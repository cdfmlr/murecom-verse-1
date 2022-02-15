package ncm

import (
	"fmt"
	"gorm.io/gorm"
	"ncm/ncmapi"
	"regexp"
	"strings"
	"sync"
)

// This file defines Models (structs) for the data collection.

// region Models

type Playlist struct {
	Id          int64  `json:"id" gorm:"primaryKey"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Tags        string `json:"tags"` // "tags":["快乐","放松","夜晚"] => "快乐,放松,夜晚"
	CoverImgUrl string `json:"coverImgUrl"`

	UpdateTime      Timestamp `json:"updateTime"`      // meta data
	TrackUpdateTime Timestamp `json:"trackUpdateTime"` // tracks

	PlayCount       int `json:"playCount"`
	SubscribedCount int `json:"subscribedCount"`

	TrackCount int `json:"trackCount"`

	Tracks []*Track `json:"tracks" gorm:"many2many:playlist_tracks"`
}

type Track struct {
	Id          int64      `json:"id" gorm:"primaryKey"`
	Name        string     `json:"name"`
	Artists     []*Artist  `json:"ar" gorm:"many2many:track_artists"`
	Pop         int        `json:"pop"`
	Album       []*Album   `json:"al" gorm:"many2many:track_albums"`
	PublishTime Timestamp  `json:"publishTime"`
	Comments    []*Comment `json:"comments" gorm:"many2many:track_comments"` // hot_comments
	Lyric       string     `json:"lyric"`                                    // 仅中文, 无则空
}

type Artist struct {
	Id   int64  `json:"id" gorm:"primaryKey"`
	Name string `json:"name"`
}

type Album struct {
	Id     int64  `json:"id" gorm:"primaryKey"`
	Name   string `json:"name"`
	PicUrl string `json:"picUrl"`
}

type Comment struct {
	Id         int64     `json:"commentId" gorm:"primaryKey"`
	User       *User     `json:"user" gorm:"foreignKey:Id"`
	Content    string    `json:"content"`
	Time       Timestamp `json:"time"`
	LikedCount int       `json:"likedCount"`
}

type User struct {
	Id       int64  `json:"userId" gorm:"primaryKey"`
	Nickname string `json:"nickname"`
}

// Timestamp 是网易云用的 JS 时间戳: 带毫秒的那种 (UNIX 时间戳+3位毫秒)
// e.g. 1642418922858
type Timestamp = int64

// endregion Models

// region JoinTable: playlist_tracks

// PlaylistTrack is a customized JoinTable for the playlist-track relationship
// with additional No. message to help Tracks list keep sorted.
//
// Help: https://gorm.io/docs/many_to_many.html#Customize-JoinTable
type PlaylistTrack struct {
	PlaylistId int64 `gorm:"primaryKey"`
	TrackId    int64 `gorm:"primaryKey"`
	No         int   // index of track in playlist
}

// BeforeSave assigns pt.No, by loading it from tmpPlaylistTrackNo.
func (pt *PlaylistTrack) BeforeSave(tx *gorm.DB) (err error) {
	pt.No = ptNoLoad(pt.PlaylistId, pt.TrackId)
	return nil
}

// endregion JoinTable: playlist_tracks

// region tmp {Playlist-Track: No} Map

// A map to store PlaylistTrack.No, to pass it into BeforeSave.
//     key  : string: keyPlaylistTrack("playlistId", "trackId")
//     value:    int: PlaylistTrack.No
// ⚠️ Do not S/L it directly, use ptNoStore and ptNoLoad instead.
var tmpPlaylistTrackNo sync.Map

// Go Tips: sync.Map 不用初始化，零值可用

// key for tmpPlaylistTrackNo
func keyPlaylistTrack(playlistId, trackId int64) string {
	return fmt.Sprint(playlistId, trackId)
}

// ptNoStore store a `no` value into tmpPlaylistTrackNo
func ptNoStore(playlistId, trackId int64, no int) {
	tmpPlaylistTrackNo.Store(keyPlaylistTrack(playlistId, trackId), no)
}

// ptNoLoad load a `no` value from tmpPlaylistTrackNo.
// Returns `no=-1` if key not exist.
func ptNoLoad(playlistId, trackId int64) (no int) {
	no = -1
	v, ok := tmpPlaylistTrackNo.LoadAndDelete(
		keyPlaylistTrack(playlistId, trackId))
	if !ok {
		return no
	}
	switch i := v.(type) {
	case int:
		no = i
	}
	return no
}

// endregion tmpPlaylistTrackNo Map

// region models conv from ncmapi

func PlaylistFromNcmapi(np *ncmapi.Playlist) *Playlist {
	p := &Playlist{
		Id:              np.Id,
		Name:            np.Name,
		Description:     np.Description,
		Tags:            tagsToString(np.Tags),
		CoverImgUrl:     np.CoverImgUrl,
		UpdateTime:      np.UpdateTime,
		TrackUpdateTime: np.TrackUpdateTime,
		PlayCount:       np.PlayCount,
		SubscribedCount: np.SubscribedCount,
		TrackCount:      np.TrackCount,
		Tracks:          []*Track{},
	}
	for i, nt := range np.Tracks {
		ptNoStore(np.Id, nt.Id, i)
		p.Tracks = append(p.Tracks, TrackFromNcmapi(&nt))
	}
	return p
}

// ["快乐","放松","夜晚"] => "快乐,放松,夜晚"
func tagsToString(tags []string) string {
	return strings.Join(tags, ",")
}

func TrackFromNcmapi(nt *ncmapi.Track) *Track {
	t := &Track{
		Id:          nt.Id,
		Name:        nt.Name,
		Artists:     []*Artist{},
		Pop:         nt.Pop,
		Album:       []*Album{AlbumFromNcmapi(&nt.Al)},
		PublishTime: nt.PublishTime,
		Comments:    []*Comment{},
	}

	for _, na := range nt.Ar {
		t.Artists = append(t.Artists, ArtistFromNcmapi(&na))
	}

	return t
}

func AlbumFromNcmapi(na *ncmapi.Album) *Album {
	a := &Album{
		Id:     na.Id,
		Name:   na.Name,
		PicUrl: na.PicUrl,
	}
	return a
}

func ArtistFromNcmapi(na *ncmapi.Artist) *Artist {
	a := &Artist{
		Id:   na.Id,
		Name: na.Name,
	}
	return a
}

func (t *Track) FillComments(comments []ncmapi.HotComment) {
	for _, nc := range comments {
		t.Comments = append(t.Comments, CommentFromNcmapi(&nc))
	}
}

func (t *Track) FillLyric(lyric *ncmapi.LyricResult) {
	s := lyricsVersionChoose(lyric)
	t.Lyric = lyricsFilter(s)
}

func CommentFromNcmapi(nc *ncmapi.HotComment) *Comment {
	c := &Comment{
		Id:         nc.CommentId,
		User:       UserFromNcmapi(&nc.User),
		Content:    nc.Content,
		Time:       nc.Time,
		LikedCount: nc.LikedCount,
	}
	return c
}

func UserFromNcmapi(nu *ncmapi.User) *User {
	u := &User{
		Id:       nu.UserId,
		Nickname: nu.Nickname,
	}
	return u
}

// endregion models conv from ncmapi

// region lyrics preprocess

// lyricsMinLen 个字以下的歌词就等于没有歌词
const lyricsMinLen = len("[99:00.00]纯音乐，请欣赏\n")

// lyricsTimelines 是歌词中时间轴的正则: [...] 或者是 (...)
var lyricsTimelines = regexp.MustCompile(`\[.*?\]|\(.*?\)`)

// lyricsFilter remove timelines
func lyricsFilter(s string) string {
	if len(s) <= lyricsMinLen { // 纯音乐，没歌词
		return ""
	}
	return lyricsTimelines.ReplaceAllLiteralString(s, "")
}

func lyricsVersionChoose(lyrics *ncmapi.LyricResult) string {
	if lyrics == nil {
		return ""
	}
	// 有翻译: 外文歌，用中文词
	if lyrics.Tlyric.Lyric != "" {
		return lyrics.Tlyric.Lyric
	}
	// 没翻译|中文歌: 用原文
	return lyrics.Lrc.Lyric
}

// endregion lyrics preprocess
