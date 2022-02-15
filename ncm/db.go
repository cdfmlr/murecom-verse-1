package ncm

import (
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var DB *gorm.DB

func InitDB() {
	var err error

	DB, err = gorm.Open(postgres.Open(Config.DB), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
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
