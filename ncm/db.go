package main

import (
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	gormlogger "gorm.io/gorm/logger"
)

var DB *gorm.DB

func InitDB() {
	var err error

	gormLogMode := gormlogger.Silent
	if Debug {
		gormLogMode = gormlogger.Error
	}

	DB, err = gorm.Open(postgres.Open(Config.DB), &gorm.Config{
		Logger: gormlogger.Default.LogMode(gormLogMode),
	})
	if err != nil {
		panic("failed to connect database: " + err.Error())
	}

	err = dbMigrate()
	if err != nil {
		panic("failed to AutoMigrate database: " + err.Error())
	}
}

func dbMigrate() error {
	err := DB.AutoMigrate(&Artist{}, &Album{}, &Comment{}, &User{}, &Track{}, &Playlist{}, &PlaylistTrack{})
	if err != nil {
		return err
	}
	err = DB.SetupJoinTable(&Playlist{}, "Tracks", &PlaylistTrack{})

	return err
}

// SavePlaylist 在一次事务中保存播放列表及其中所有曲目，以及各种全部信息
func SavePlaylist(playlist *Playlist) {
	DB.Save(playlist)
}

func PlaylistExist(pid int64) bool {
	var p Playlist
	DB.Take(&p, pid)
	return p.Id != 0
}
