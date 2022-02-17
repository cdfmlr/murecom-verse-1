package main

import (
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
	PlaylistWorksTimeoutSec = 320
	TrackWorksGoroutine     = 16
	TrackWorksTimeoutSec    = 160
)

var ProgressCount = 1

// TODO: /top/playlist 还支持各种分类的，可以支持一下
//  https://binaryify.github.io/NeteaseCloudMusicApi/#/?id=歌单-网友精选碟-

// FetchTopPlaylists request TopPlaylists API, yields ncmapi.Playlist.
// 一次给一个, bufferSize=TopPlaylistPagingLimit, 完了 close
func FetchTopPlaylists(client ncmapi.Client, catalog string) <-chan ncmapi.Playlist {
	ch := make(chan ncmapi.Playlist, TopPlaylistBufferSize)

	go func() {
		total := TopPlaylistPagingLimit + 1 // 所有播放列表的数量，一开始随便设个值，从第一次请求中取真值
		count := 0                          // 已获取计数
		for count < total {
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
		close(ch)
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

// FetchTracks 只给一次，失败给零值，不 close，调用者自行判断
func FetchTracks(client ncmapi.Client, pid int64, trackCount int) <-chan []ncmapi.Track {
	ch := make(chan []ncmapi.Track)

	go func() {
		count := 0
		var tracks []ncmapi.Track

		for offset := 0; offset*TracksPagingLimit < trackCount; offset++ {
			ptr, err := client.PlaylistTracks(pid, TracksPagingLimit, offset)
			if err != nil {
				logger.Error("FetchTracks failed: ", fmt.Sprintf("pid=%v, err=%v", pid, err.Error()))
				close(ch)
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

// FetchLyrics 只给一次，失败给零值，不 close，调用者自行判断
func FetchLyrics(client ncmapi.Client, tid int64) <-chan ncmapi.LyricResult {
	ch := make(chan ncmapi.LyricResult)

	go func() {
		lr, err := client.Lyric(tid)
		if err != nil {
			logger.Warn("FetchLyrics failed, use zero value:", fmt.Sprintf("tid=%v, err=%v", tid, err.Error()))
			slowdown()
		}
		ch <- *lr
	}()

	return ch
}

// FetchComments 只给一次，失败给零值，不 close，调用者自行判断
func FetchComments(client ncmapi.Client, tid int64) <-chan []ncmapi.HotComment {
	ch := make(chan []ncmapi.HotComment)

	go func() {
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
func PlaylistWorks(client ncmapi.Client, np *ncmapi.Playlist) error {
	logger.Debug(fmt.Sprintf("PlaylistWorks: playlist: id=%v, name=%v", np.Id, np.Name))
	// Fetch all tracks
	tracks := make([]ncmapi.Track, np.TrackCount)
	trackCount := np.TrackCount

	if trackCount >= Config.MaxTracks {
		logger.Warn(fmt.Sprintf("Looong playlist (%v: %v): cut to maxLen=%v", np.Id, np.Name, Config.MaxTracks))
		trackCount = Config.MaxTracks
	}
	select {
	case tracks = <-FetchTracks(client, np.Id, trackCount):
		// Do nothing here: break select
	case <-time.After(PlaylistWorksTimeoutSec * time.Second):
		err := errors.New("PlaylistWorks.FetchTracks: timeout, cancel")
		logger.Error(err.Error())
		return err
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
		wg.Add(1)
		go func() {
			_ = TrackWorks(client, t)
			wg.Done()
		}()

		if i%TrackWorksGoroutine == 0 {
			wg.Wait()
		}
		slowdown()
	}

	wg.Wait()

	// All fetching works done, save it.
	SavePlaylist(p)

	logger.Debug(fmt.Sprintf("SavePlaylist: id=%v, name=%v", p.Id, p.Name))

	return nil
}

// TrackWorks 完成 FetchLyrics, FetchComments，原址填写 *Track t
func TrackWorks(client ncmapi.Client, t *Track) error {
	logger.Debug(fmt.Sprintf("TrackWorks: track: id=%v, name=%v", t.Id, t.Name))

	if t.Id == 0 { // ??? something wrong
		err := errors.New(fmt.Sprint("unexpected: TrackWorks on zero Track, skip: ", t))
		logger.Error(err.Error())
		return err
	}

	// 开始子任务
	chLyrics := FetchLyrics(client, t.Id)
	chComments := FetchComments(client, t.Id)

	for i := 0; i < 2; i++ {
		select {
		case lyrics := <-chLyrics:
			t.FillLyric(&lyrics)
		case comments := <-chComments:
			t.FillComments(comments)
		case <-time.After(TrackWorksTimeoutSec * time.Second): // Timeout
			// TODO: Add context to cancel goroutines
			slowdown()
			err := errors.New("TrackWorks(lyrics, comments): timeout, use zero value")
			logger.Warn(err.Error())
			return err
		}
	}

	return nil
}

// Task 统筹安排从 FetchTopPlaylists 到 PlaylistWorks 到 TracksWorks 的一系列工作，
// 完成对 catalog 分类的爬取工作。
func Task(catalog string) {
	// 计数器、最大值
	max := Config.MaxPlaylists
	if max >= 100 {
		ProgressCount = max / 100
	}

	// log tasks
	logger.Info(fmt.Sprintf("NCM Task: catalogs=%q, max_playlists=%v (±%v)",
		catalog, max, TopPlaylistBufferSize))

	// 结束循环
	chDone := make(chan struct{}, 1)
	// 暂停等待
	chWait := make(chan struct{}, 1)

	// 更新计数器，打印进度
	chCount := make(chan struct{}, TopPlaylistBufferSize)
	go func() {
		count := 0 // 已获取计数
		for range chCount {
			count += 1
			if count%ProgressCount == 0 {
				logProgress(count, max)
				chWait <- struct{}{}
			}
			if count >= max {
				chDone <- struct{}{}
			}
		}
	}()

	// 根客户端，取列表
	client := ncmapi.GetClient()
	if client == nil {
		logger.Error("Get Client to FetchTopPlaylists failed, exit")
		return
	}
	chPlaylist := FetchTopPlaylists(client, Config.Catalogs[0])

	// 填充列表、保存
	wg := sync.WaitGroup{}

LOOP:
	for { // 这里取 count 可能略脏，但多几个少几个似乎也没关系
		select {
		case <-chDone:
			logger.Info("task ", catalog, ": count=max: done.")
			break LOOP
		case <-chWait:
			wg.Wait()
		default:
			slowdown()
		}

		np, ok := <-chPlaylist
		if !ok {
			logger.Warn("task ", catalog, ": no more playlists, done.")
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
			err := PlaylistWorks(c, &np)

			if err != nil {
				wg.Done()
				return
			}
			chCount <- struct{}{}
			wg.Done()
		}()
	}

	wg.Wait()
	close(chCount)
}

func Master() {
	logger.Info(fmt.Sprintf(
		"NCM Master Tasks:\n\t catalogs=%q\n\t max %v playlists for each catalog.\n\t Good luck!",
		Config.Catalogs, Config.MaxPlaylists))

	for _, catalog := range Config.Catalogs {
		Task(catalog)
	}

	logger.Info("NCM Master: Done.")
}

func logProgress(count, max int) {
	logger.Info(fmt.Sprintf(
		"progress %v%%: got %v/%v playlists.",
		count*100/max, count, max))
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
