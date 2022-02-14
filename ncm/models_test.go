package ncm

import (
	"encoding/json"
	"ncm/ncmapi"
	"os"
	"testing"
)

// TODO: test modelsconv

func _loadTestConfig() (config map[string]string) {
	config = map[string]string{}
	file, err := os.Open("test_config.json")
	if err != nil {
		panic(err)
	}
	err = json.NewDecoder(file).Decode(&config)
	if err != nil {
		panic(err)
	}
	return config
}

// 测试转化函数的同时，也测试 ncmapi 包外部访问。
// 还作为一的 Demo, 获取并转化一个播放列表。
func TestPlaylistFromNcmapi(t *testing.T) {
	// ncmapi Init
	t.Log("🚚 ncmapi.Init")
	config := _loadTestConfig()
	err := ncmapi.Init([]ncmapi.ClientConfig{{
		Phone:       config["phone"],
		PasswordMD5: config["md5_password"],
		Server:      config["server"]}}, nil)
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
		ptr, err := client.PlaylistTracks(int(np.Id), limit, 0)
		if err != nil {
			t.Errorf("Get PlaylistTracks failed (limit=%v, offset=%v): %v", limit, offset, err)
		}
		np.Tracks = append(np.Tracks, ptr.Songs...)
	}

	// local model
	playlist := PlaylistFromNcmapi(&np)

	// Lyrics
	t.Log("🚚 Lyrics")
	for _, tk := range playlist.Tracks {
		lr, err := client.Lyric(int(tk.Id))
		if err != nil {
			t.Errorf("Get lyrics (%v: %v) failed: %v", tk.Id, tk.Name, err)
		}
		tk.FillLyric(lr)
	}

	// Comments
	t.Log("🚚 Comments")
	for _, tk := range playlist.Tracks {
		cr, err := client.TrackHotComment(int(tk.Id))
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
	t.Log("✅ result playlist:", fname)
}
