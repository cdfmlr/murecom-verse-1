package main

import (
	"fmt"
	"github.com/cdfmlr/murecom-verse-1/ncm/ncmapi"
	"net/http"
	_ "net/http/pprof"
)

const Debug = false

func main() {
	// init
	InitConfig("config.json")

	// ncmapi Init
	err := ncmapi.Init(NcmapiConfigs(), ncmapiLogger{logger})
	if err != nil {
		logger.Fatal("ncmapi Init failed:", err)
	}
	InitDB()
	InitFetcher()

	// profile
	if Config.Profile != "" {
		// http://{ip:port}/debug/pprof/
		go func() {
			logger.Info(fmt.Sprintf("Profile: http://%v/debug/pprof", Config.Profile))
			panic(http.ListenAndServe(Config.Profile, nil))
		}()
	}

	// run
	Master()
}
