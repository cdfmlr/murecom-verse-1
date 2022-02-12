package ncm

import (
	"ncm/ncmapi"
	"reflect"
	"testing"
)

// TODO: test modelsconv

func TestPlaylistFromNcmapi(t *testing.T) {
	type args struct {
		np *ncmapi.Playlist
	}
	tests := []struct {
		name string
		args args
		want *Playlist
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PlaylistFromNcmapi(tt.args.np); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("PlaylistFromNcmapi() = %v, want %v", got, tt.want)
			}
		})
	}
}
