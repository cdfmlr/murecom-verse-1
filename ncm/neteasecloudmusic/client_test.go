package neteasecloudmusic

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"testing"
)

var testClient *client

func init() {
	config := map[string]string{}
	file, err := os.Open("test_config.json")
	if err != nil {
		panic(err)
	}
	err = json.NewDecoder(file).Decode(&config)
	if err != nil {
		panic(err)
	}
	testClient, err = newClient(config["phone"], config["md5_password"], config["server"])
	if err != nil {
		panic(err)
	}
	fmt.Printf("testClient: %#v\n", testClient)
}

// region Test_client_login

// 这个接口不太一样，要自己实现一下 Resulter 接口，提供正确的 success 方法。
// 它响应多套了一层: {data: { code: 200, ...} }
type loginStatus4TestOnly map[string]interface{}

func (l loginStatus4TestOnly) success() bool {
	// MAGIC! DO NOT TOUCH! 我困了，随便写的，不要在意，反正这样能工作
	// fmt.Println(">>>>>", l["data"].(map[string]interface{})["code"], int(l["data"].(map[string]interface{})["code"].(float64)) == CodeOK)
	return int(l["data"].(map[string]interface{})["code"].(float64)) == CodeOK
}

func Test_client_login(t *testing.T) {
	c := testClient
	status := loginStatus4TestOnly{}
	err := c.requestAPI(c.get, c.apiUrl("/login/status"), url.Values{}, &status)
	if err != nil {
		t.Error("❌", err)
	}
	t.Logf("✅ %#v", status)
}

// endregion Test_client_login

func Test_client_Lyric(t *testing.T) {
	type args struct {
		id int
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"Lyrics_zh_cn_song", args{id: 33894312}, false}, // 情非得已
		{"Lyrics_pure_music", args{id: 27785067}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testClient
			got, err := c.Lyric(tt.args.id)
			if (err != nil) != tt.wantErr {
				t.Errorf("Lyric() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			t.Log("✅", got)
		})
	}
}

func Test_client_PlaylistDetail(t *testing.T) {
	type args struct {
		id int64
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"playlistDetail", args{id: 611441282}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testClient
			got, err := c.PlaylistDetail(tt.args.id)
			if (err != nil) != tt.wantErr {
				t.Errorf("PlaylistDetail() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			t.Log("✅", got)
		})
	}
}

func Test_client_PlaylistTracks(t *testing.T) {

	type args struct {
		id     int
		limit  int
		offset int
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"playlistTracks", args{
			id:     611441282,
			limit:  50,
			offset: 0,
		}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testClient
			got, err := c.PlaylistTracks(tt.args.id, tt.args.limit, tt.args.offset)
			if (err != nil) != tt.wantErr {
				t.Errorf("PlaylistTracks() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			t.Log("✅", got)
			t.Log("✅ len songs: ", len(got.Songs))
		})
	}
}

func Test_client_TopPlaylists(t *testing.T) {

	type args struct {
		limit  int
		offset int
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"topPlaylists", args{limit: 50, offset: 0}, false},
		{"topPlaylists_p2", args{limit: 50, offset: 50}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testClient
			got, err := c.TopPlaylists(tt.args.limit, tt.args.offset)
			if (err != nil) != tt.wantErr {
				t.Errorf("TopPlaylists() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			var names []string
			for _, p := range got.Playlists {
				names = append(names, p.Name)
			}
			t.Logf("✅got.Playlists.names: %#v", names)
			t.Log("✅len(got.Playlists)=", len(got.Playlists))
			t.Log("✅got.Total=", got.Total)
		})
	}
}

func Test_client_TrackHotComment(t *testing.T) {

	type args struct {
		id int
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"hotcomment", args{id: 33894312}, false}, // 情非得已
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testClient
			got, err := c.TrackHotComment(tt.args.id)
			if (err != nil) != tt.wantErr {
				t.Errorf("TrackHotComment() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			t.Log("✅len(got.HotComments)", len(got.HotComments))
			var comments []string
			for _, c := range got.HotComments {
				comments = append(comments, fmt.Sprintf("[👍%v] %v", c.LikedCount, c.Content))
			}

			t.Logf("✅%#v", comments)
		})
	}
}
