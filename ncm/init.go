package ncm

import "github.com/cdfmlr/murecom-verse-1/ncm/ncmapi"

func InitAll(configfile string) {
	// init
	InitConfig(configfile)

	// ncmapi Init
	err := ncmapi.Init(NcmapiConfigs(), ncmapiLogger{logger})
	if err != nil {
		logger.Fatal("ncmapi Init failed:", err)
	}
	InitDB()
	InitFetcher()
}
