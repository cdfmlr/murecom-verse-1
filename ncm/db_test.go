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
