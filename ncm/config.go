package main

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"ncm/ncmapi"
	"os"
)

var Config struct {
	NcmClient []struct {
		Phone       string `json:"phone"`
		PasswordMD5 string `json:"password_md5"`
		Server      string `json:"server"`
	} `json:"client"`
	DB            string   `json:"db"`
	MaxPlaylists  int      `json:"max_playlists"` // 对 Catalogs 中的每一项最多爬 MaxPlaylists 张: default 1000
	MaxTracks     int      `json:"max_tracks"`    // 一张播放列表中最多爬几首歌，默认(≤0): 不限制(MaxInt)，但一张列表最多搞 10 分钟，搞不完会 cancel 掉
	Speed         int      `json:"speed"`         // ±200, 正慢负快
	Profile       string   `json:"profile"`       // 放 profile 结果的目录
	Catalogs      []string `json:"catalogs"`      // 若为空，遍历所有已知的: ncmapi.AllTopPlaylistsCatalogs
	ErrorHandling struct {
		SleepSec   int `json:"sleep_sec"`   // 出错了停下来等几秒，ncm api server 可能要重启, default 3
		MaxRetries int `json:"max_retries"` // 重试次数，default 5
	} `json:"error_handling"`
}

// InitConfig read the config file
func InitConfig(file string) {
	f, err := os.Open(file)
	must(err)
	j, err := ioutil.ReadAll(f)
	must(err)
	err = json.Unmarshal(j, &Config)
	must(err)

	configDefault()
}

// 设定默认值
func configDefault() {

	if Config.MaxPlaylists <= 0 {
		Config.MaxPlaylists = 1000
	}
	if Config.MaxTracks <= 0 {
		Config.MaxTracks = math.MaxInt
	}
	if Config.Speed < -200 { // 下限
		Config.Speed = -200
	}
	if len(Config.Catalogs) == 0 {
		logger.Info("Config: no catalogs specified, use all known: ", ncmapi.AllTopPlaylistsCatalogs)
		Config.Catalogs = ncmapi.AllTopPlaylistsCatalogs
	}
	if Config.ErrorHandling.MaxRetries <= 0 {
		Config.ErrorHandling.MaxRetries = 5
	}
	if Config.ErrorHandling.SleepSec <= 0 {
		Config.ErrorHandling.SleepSec = 3
	}
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
