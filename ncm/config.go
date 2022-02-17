package main

import (
	"encoding/json"
	"io/ioutil"
	"ncm/ncmapi"
	"os"
)

var Config struct {
	NcmClient []struct {
		Phone       string `json:"phone"`
		PasswordMD5 string `json:"password_md5"`
		Server      string `json:"server"`
	} `json:"client"`
	DB           string `json:"db"`
	MaxPlaylists int    `json:"max_playlists"`
	Speed        int    `json:"speed"` // ±200
	Profile      string `json:"profile"`
}

// InitConfig read the config file
func InitConfig(file string) {
	f, err := os.Open(file)
	must(err)
	j, err := ioutil.ReadAll(f)
	must(err)
	err = json.Unmarshal(j, &Config)
	must(err)
}

// panic if error
func must(err error) {
	if err != nil {
		panic("failed to load config: " + err.Error())
	}
}

func NcmapiConfigs() []ncmapi.ClientConfig {
	var configs []ncmapi.ClientConfig
	for _, c := range Config.NcmClient {
		configs = append(configs, ncmapi.ClientConfig{
			Phone:       c.Phone,
			PasswordMD5: c.PasswordMD5,
			Server:      c.Server,
		})
	}
	return configs
}
