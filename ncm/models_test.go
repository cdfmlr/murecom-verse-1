package main

import (
	"encoding/json"
	"github.com/cdfmlr/murecom-verse-1/ncm/ncmapi"
	"os"
	"testing"
)

// 测试转化函数的同时，也测试 ncmapi 包外部访问。
// 还作为一的 Demo, 获取并转化一个播放列表。
func TestPlaylistFromNcmapi(t *testing.T) {
	InitConfig("test_config.json")
	config := Config.NcmClient[0]
	if config.Server == "" {
		t.Fatal("❌ No config!")
	}

	// ncmapi Init
	t.Log("🚚 ncmapi.Init")
	err := ncmapi.Init([]ncmapi.ClientConfig{{
		Phone:       config.Phone,
		PasswordMD5: config.PasswordMD5,
		Server:      config.Server}}, nil)
	if err != nil {
		t.Fatal("ncmapi Init failed:", err)
	}

	// GetClient
	t.Log("🚚 GetClient")
	client := ncmapi.GetClient()
	if client == nil {
		t.Fatal("GetClient failed: client==nil")
	}

	// TopPlaylist
	t.Log("🚚 TopPlaylist")
	tpr, err := client.TopPlaylists(1, 0)
	if err != nil {
		t.Fatal("Get TopPlaylists failed:", err)
	}

	// PlaylistDetail
	t.Log("🚚 PlaylistDetail")
	p := &tpr.Playlists[0]
	pdr, err := client.PlaylistDetail(p.Id)
	if err != nil {
		t.Fatal("Get PlaylistDetail failed:", err)
	}
	np := pdr.Playlist

	// Tracks
	t.Log("🚚 Tracks")
	np.Tracks = []ncmapi.Track{}
	limit := 50
	for offset := 0; offset*limit < np.TrackCount; offset++ {
		ptr, err := client.PlaylistTracks(np.Id, limit, offset)
		if err != nil {
			t.Errorf("Get PlaylistTracks failed (limit=%v, offset=%v): %v", limit, offset, err)
		}

		// 它每次一定会给满 limit 个数据，最后一页会给倒数 limit 个
		// 所以可能和上一页重复，需要计算一下，重新切个片
		// 例如 总共 26 条, limit=10:
		//   req(limit=10, offset=0): 0~9
		//   req(limit=10, offset=1): 10~19
		//   req(limit=10, offset=2): 16~25

		got := len(np.Tracks) + len(ptr.Songs)
		remind := np.TrackCount - got

		start := 0
		if remind < 0 { // 最后一页：会有重复数据
			start = len(ptr.Songs) + remind
		}
		np.Tracks = append(np.Tracks, ptr.Songs[start:]...)
	}

	// region Tracks tests

	if np.TrackCount != len(np.Tracks) {
		t.Errorf("❌ TrackCount=%v, len(Tracks)=%v", np.TrackCount, len(np.Tracks))
	}
	trackSet := make(map[int64]struct{})
	for _, tk := range np.Tracks {
		if _, exist := trackSet[tk.Id]; exist {
			t.Errorf("❌ Duplicated track: %v - %v", tk.Id, tk.Name)
		}
		trackSet[tk.Id] = struct{}{}
	}
	if len(trackSet) != len(np.Tracks) {
		t.Errorf("❌ len(Tracks)=%v, len(tracksSet)=%v", len(np.Tracks), len(trackSet))
	}

	// endregion Tracks tests

	// local model
	playlist := PlaylistFromNcmapi(&np)

	// Lyrics
	t.Log("🚚 Lyrics")
	for _, tk := range playlist.Tracks {
		lr, err := client.Lyric(tk.Id)
		if err != nil {
			t.Errorf("Get lyrics (%v: %v) failed: %v", tk.Id, tk.Name, err)
		}
		tk.FillLyric(lr)
	}

	// Comments
	t.Log("🚚 Comments")
	for _, tk := range playlist.Tracks {
		cr, err := client.TrackHotComment(tk.Id)
		if err != nil {
			t.Errorf("Get TrackHotComment (%v: %v) failed: %v", tk.Id, tk.Name, err)
		}
		tk.FillComments(cr.HotComments)
	}

	// Done!
	// 结果太多了，写到个 json 文件里方便检查。。
	fname := "models_test_result.json"
	result, _ := json.Marshal(playlist)
	_ = os.WriteFile(fname, result, 0600)
	t.Logf("✅ result playlist @ %#v: name=%#v, tracks=%v.", fname, playlist.Name, playlist.TrackCount)
	var trackNames []string
	for _, tk := range playlist.Tracks {
		trackNames = append(trackNames, tk.Name)
	}
	t.Logf("✅ tracks: %#v", trackNames)
}
