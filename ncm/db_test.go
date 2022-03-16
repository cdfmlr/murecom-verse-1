package ncm

import (
	"encoding/json"
	"os"
	"testing"
)

func TestInitDB(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		InitConfig("test_config.json")
		InitDB()
	})
	t.Run("Save", func(t *testing.T) {
		playlist := Playlist{}

		file, err := os.Open("models_test_result.json")
		if err != nil {
			panic(err)
		}
		err = json.NewDecoder(file).Decode(&playlist)
		if err != nil {
			panic(err)
		}

		for i, tk := range playlist.Tracks {
			ptNoStore(playlist.Id, tk.Id, i)
		}

		DB.Save(&playlist)
	})
}

func TestPlaylistExist(t *testing.T) {
	InitConfig("test_config.json")
	InitDB()

	type args struct {
		pid int64
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{"exist", args{pid: 2862916340}, true},
		{"no-exist", args{pid: 99999999999}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PlaylistExist(tt.args.pid); got != tt.want {
				t.Errorf("PlaylistExist() = %v, want %v", got, tt.want)
			}
		})
	}
}
