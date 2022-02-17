package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"ncm/ncmapi"
	"sync"
	"time"
)

const (
	TopPlaylistPagingLimit  = 35
	TopPlaylistBufferSize   = 10
	TracksPagingLimit       = 50
	PlaylistWorksTimeoutSec = 600
	TrackWorksTimeoutSec    = 60
	TrackWorksGoroutine     = 12
	TaskWorksGap            = 10 // 每弄完这么多的列表，停下来休息一下
)

// FetchTopPlaylists request TopPlaylists API, yields ncmapi.Playlist.
// 一次给一个, bufferSize=TopPlaylistPagingLimit, 完了 close
func FetchTopPlaylists(ctx context.Context, client ncmapi.Client, catalog string) <-chan ncmapi.Playlist {
	ch := make(chan ncmapi.Playlist, TopPlaylistBufferSize)

	go func() {
		defer close(ch)

		total := TopPlaylistPagingLimit + 1 // 所有播放列表的数量，一开始随便设个值，从第一次请求中取真值
		count := 0                          // 已获取计数
		for count < total {
			select {
			case <-ctx.Done():
				logger.Debug("FetchTopPlaylists canceled, close chan and return early")
				return
			default:
			}
			// Request API
			tpr, err := client.TopPlaylistsCatalog(catalog, TopPlaylistPagingLimit, count) // 这个的 offset 是几就是从第几个开始
			if err != nil {
				logger.Warn("FetchTopPlaylists failed, skip:", err)
				continue
			}

			// Yield playlists
			for _, p := range tpr.Playlists {
				ch <- p
				count++
			}

			total = tpr.Total
		}
		// no more data to yield
	}()

	return ch
}

// Deprecated: FetchPlaylistDetail 这个似乎没用，需要的信息好像 TopPlaylists 就返回了。
func FetchPlaylistDetail(client ncmapi.Client, pid int64) <-chan ncmapi.Playlist {
	ch := make(chan ncmapi.Playlist)

	go func() {
		pdr, err := client.PlaylistDetail(pid)
		if err != nil {
			logger.Error("FetchPlaylistDetail error, close pipe:", err)
			close(ch)
		}

		ch <- pdr.Playlist
	}()

	return ch
}

// FetchTracks 只给一次，失败给零值
func FetchTracks(ctx context.Context, client ncmapi.Client, pid int64, trackCount int) <-chan []ncmapi.Track {
	ch := make(chan []ncmapi.Track)

	go func() {
		defer close(ch)
		count := 0
		var tracks []ncmapi.Track

		for offset := 0; offset*TracksPagingLimit < trackCount; offset++ {
			select {
			case <-ctx.Done():
				logger.Debug("FetchTracks canceled, close chan and return early")
				return
			default:
			}

			ptr, err := client.PlaylistTracks(pid, TracksPagingLimit, offset)
			if err != nil {
				logger.Error("FetchTracks failed: ", fmt.Sprintf("pid=%v, err=%v", pid, err.Error()))
				return
			}

			// 它每次一定会给满 limit 个数据，最后一页会给倒数 limit 个
			// 所以可能和上一页重复，需要计算一下，重新切个片
			// 例如 总共 26 条, limit=10:
			//   req(limit=10, offset=0): 0~9
			//   req(limit=10, offset=1): 10~19
			//   req(limit=10, offset=2): 16~25

			got := count + len(ptr.Songs)
			remind := trackCount - got

			start := 0
			if remind < 0 { // 最后一页：会有重复数据
				start = len(ptr.Songs) + remind
			}
			tracks = append(tracks, ptr.Songs[start:]...)
			count += len(ptr.Songs)
		}
		ch <- tracks
	}()

	return ch
}

// FetchLyrics 只给一次，失败给零值
func FetchLyrics(ctx context.Context, client ncmapi.Client, tid int64) <-chan ncmapi.LyricResult {
	ch := make(chan ncmapi.LyricResult)

	go func() {
		defer close(ch)

		select {
		case <-ctx.Done():
			return
		default:
		}
		lr, err := client.Lyric(tid)
		if err != nil {
			logger.Warn("FetchLyrics failed, use zero value:", fmt.Sprintf("tid=%v, err=%v", tid, err.Error()))
			slowdown()
		}
		ch <- *lr
	}()

	return ch
}

// FetchComments 只给一次，失败给零值
func FetchComments(ctx context.Context, client ncmapi.Client, tid int64) <-chan []ncmapi.HotComment {
	ch := make(chan []ncmapi.HotComment)

	go func() {
		defer close(ch)

		select {
		case <-ctx.Done():
			return
		default:
		}
		cr, err := client.TrackHotComment(tid)
		if err != nil {
			logger.Warn("FetchComments failed, use zero value:", fmt.Sprintf("tid=%v, err=%v", tid, err.Error()))
			slowdown()
		}
		ch <- cr.HotComments
	}()

	return ch
}

// PlaylistWorks 从 FetchTopPlaylists 拿到的 ncmapi.Playlist 开始，
// 做完一套 FetchTracks, FetchLyrics, FetchComments，
// 转化出的最终 Playlist，并保存。
func PlaylistWorks(ctx context.Context, client ncmapi.Client, np *ncmapi.Playlist) error {
	logger.Debug(fmt.Sprintf("PlaylistWorks: playlist: id=%v, name=%v", np.Id, np.Name))
	// Fetch all tracks
	tracks := make([]ncmapi.Track, np.TrackCount)
	trackCount := np.TrackCount

	if trackCount >= Config.MaxTracks {
		logger.Warn(fmt.Sprintf("Looong playlist (%v: %v): cut to maxLen=%v", np.Id, np.Name, Config.MaxTracks))
		trackCount = Config.MaxTracks
	}

	// context 加上 Timeout，超时快停
	ctx, cancel := context.WithTimeout(ctx, PlaylistWorksTimeoutSec*time.Second)
	defer cancel()

	select {
	case <-ctx.Done():
		logger.Debug("PlaylistWorks=>FetchTracks: context canceled, return early:", ctx.Err())
		return errors.New("context canceled: " + ctx.Err().Error())
	case tracks = <-FetchTracks(ctx, client, np.Id, trackCount):
		// Do nothing here: break select
	}

	// 整理结果
	if len(tracks) == 0 {
		err := errors.New("PlaylistWorks: got no tracks, do not save")
		slowdown()
		logger.Error(err.Error())
		return err
	}

	// to Playlist
	np.Tracks = tracks
	p := PlaylistFromNcmapi(np)

	if p == nil || p.Id == 0 { // ??? something wrong
		err := errors.New(fmt.Sprint("Unexpected: PlaylistWorks got zero PlaylistFromNcmapi, do not save: playlist=", p.Id))
		logger.Error(err.Error())
		return err
	}

	// lyrics and comments for each track
	wg := sync.WaitGroup{}

	for i, t := range p.Tracks {
		select {
		case <-ctx.Done():
			logger.Debug("PlaylistWorks=>TrackWorks: context canceled, returns early: ", ctx.Err().Error())
			return errors.New("context canceled" + ctx.Err().Error())
		default:
			if i%TrackWorksGoroutine == 0 {
				wg.Wait()
			} else {
				slowdown()
			}
		}

		wg.Add(1)
		t := t
		go func() {
			_ = TrackWorks(ctx, client, t)
			wg.Done()
		}()
	}

	wg.Wait()

	// 到了这边就不管 context done 不 done 了，工作都做完了，就保存了呗。

	// All fetching works done, save it.
	SavePlaylist(p)

	logger.Debug(fmt.Sprintf("SavePlaylist: id=%v, name=%v", p.Id, p.Name))

	return nil
}

// TrackWorks 完成 FetchLyrics, FetchComments，原址填写 *Track t
func TrackWorks(ctx context.Context, client ncmapi.Client, t *Track) error {
	logger.Debug(fmt.Sprintf("TrackWorks: track: id=%v, name=%v", t.Id, t.Name))

	if t.Id == 0 { // ??? something wrong
		err := errors.New(fmt.Sprint("unexpected: TrackWorks on zero Track, skip: ", t))
		logger.Error(err.Error())
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, TrackWorksTimeoutSec*time.Second)
	defer cancel()

	// 开始子任务
	chLyrics := FetchLyrics(ctx, client, t.Id)
	chComments := FetchComments(ctx, client, t.Id)

	for i := 0; i < 2; i++ {
		select {
		case <-ctx.Done():
			logger.Debug("TrackWorks: context canceled, returns early: ", ctx.Err().Error())
			return errors.New("context canceled" + ctx.Err().Error())
		case lyrics := <-chLyrics:
			t.FillLyric(&lyrics)
		case comments := <-chComments:
			t.FillComments(comments)
		}
	}

	return nil
}

// Task 统筹安排从 FetchTopPlaylists 到 PlaylistWorks 到 TracksWorks 的一系列工作，
// 完成对 catalog 分类的爬取工作。
// n: 第几个工作，为了打日志好看
func Task(catalog string, n int) {
	// 最大值，设置打印进度
	max := Config.MaxPlaylists

	// log tasks
	logger.Info(fmt.Sprintf("NCM Task %v: catalogs=%q, max_playlists=%v (±%v)",
		n, catalog, max, TopPlaylistBufferSize))

	// context to cancel works
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 计数器
	chCount := make(chan string, TopPlaylistBufferSize)
	defer close(chCount)

	// 暂停等待
	chWait := make(chan struct{}, 1)

	// 计数器工作
	go func() {
		count := 0 // 已获取计数
		for s := range chCount {
			count += 1
			logTaskProgress(catalog, count, max, n, s)

			if count >= max {
				logger.Info(fmt.Sprintf("task (%q): count=max: cancel contexts and return", catalog))
				cancel()
			} else if count%TaskWorksGap == 0 {
				logger.Debug(fmt.Sprintf("TaskWorksGap (after %v playlists): wait works done and have a rest.", TaskWorksGap))
				chWait <- struct{}{}
			}
		}
	}()

	// 根客户端，取列表
	client := ncmapi.GetClient()
	if client == nil {
		logger.Error("Get Client to FetchTopPlaylists failed, exit")
		return
	}
	chPlaylist := FetchTopPlaylists(ctx, client, catalog)

	// 填充列表、保存
	wg := sync.WaitGroup{}

LOOP:
	for { // 这里取 count 可能略脏，但多几个少几个似乎也没关系
		select {
		case <-ctx.Done():
			return
		case <-chWait:
			wg.Wait()
			for i := 0; i < 3; i++ {
				slowdown()
			}
		default:
			slowdown()
		}

		np, ok := <-chPlaylist
		if !ok {
			logger.Warn("task ", catalog, ": no more playlists, wait playlist works done.")
			break LOOP
		}

		logger.Debug(fmt.Sprintf("Master: playlist: id=%v, name=%v", np.Id, np.Name))

		// Seen?
		if PlaylistExist(np.Id) {
			logger.Debug("PlaylistExist, continue. pid=", np.Id)
			continue
		}

		wg.Add(1)
		go func() {
			c := ncmapi.GetClient()
			if c == nil {
				c = client
			}
			err := PlaylistWorks(ctx, c, &np)

			if err == nil { // success
				msg := fmt.Sprintf("%v <%v>", np.Id, np.Name)
				chCount <- msg
			}
			wg.Done()
		}()
	}

	wg.Wait()
}

func Master() {
	logger.Info(fmt.Sprintf(
		"NCM Master Tasks:\n\t catalogs=%q\n\t max %v playlists for each catalog.\n\t Good luck!",
		Config.Catalogs, Config.MaxPlaylists))

	for i, catalog := range Config.Catalogs {
		Task(catalog, i)

		//logger.Info(fmt.Sprintf("NCM Master: %v%% (%v/%v) tasks done.",
		//	(i+1)*100/len(Config.Catalogs), i+1, len(Config.Catalogs)))
		logger.Progress(i+1, len(Config.Catalogs),
			fmt.Sprintf("NCM Master: %v/%v tasks done.", i+1, len(Config.Catalogs)))
	}
}

func logTaskProgress(catalog string, count, max int, task int, msg string) {
	//p := (task*max + count) * 100 / (max * len(Config.Catalogs))
	//logger.Info(fmt.Sprintf(
	//	"%v%%: got playlist: %v (%v/%v of %q)",
	//	p, msg, count, max, catalog))
	logger.Progress(task*max+count, max*len(Config.Catalogs),
		fmt.Sprintf("got playlist: %v (%v/%v of %q)", msg, count, max, catalog))
}

func slowdown() {
	speed := 200 + Config.Speed
	if speed < 0 {
		speed = 200
	}
	time.Sleep(time.Duration(speed) * time.Millisecond)
	if rand.Int31n(10) < 4 {
		time.Sleep(300 * time.Millisecond)
	}
	if Debug {
		logger.Info("Debug: slowdown")
		time.Sleep(time.Second)
	}
}
