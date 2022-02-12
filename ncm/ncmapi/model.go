package ncmapi

type Resulter interface {
	success() bool
}

// Result 似乎是所有接口都会返回有的一个值
// TODO: 也许该给所有结构体加上 Result 的嵌入，目前只有 XxxResult 嵌入了 Result
type Result struct {
	Code int `json:"code"`
}

const CodeOK = 200

func (r Result) success() bool {
	return r.Code == CodeOK
}

// region API results: 各种 API 请求返回的东西

// XXX: 注意那些我样本里面 null 或者是 {} 的，类型给留了 interface{}
//      这种样子的东西解析出来是默认的，可能是数组、或者 map 什么的

// LoginResult 手机号登录 /login/cellphone
type LoginResult struct {
	Result
	LoginType int     `json:"loginType"`
	Account   Account `json:"account"`
	Token     string  `json:"token"`
	Profile   Profile `json:"profile"`
	Bindings  []struct {
		UserId       int64  `json:"userId"`
		Url          string `json:"url"`
		Expired      bool   `json:"expired"`
		BindingTime  int64  `json:"bindingTime"`
		TokenJsonStr string `json:"tokenJsonStr"`
		ExpiresIn    int    `json:"expiresIn"`
		RefreshTime  int    `json:"refreshTime"`
		Id           int64  `json:"id"`
		Type         int    `json:"type"`
	} `json:"bindings"`
	Cookie string `json:"cookie"`
}

type Account struct {
	Id                 int64  `json:"id"`
	UserName           string `json:"userName"`
	Type               int    `json:"type"`
	Status             int    `json:"status"`
	WhitelistAuthority int    `json:"whitelistAuthority"`
	CreateTime         int64  `json:"createTime"`
	Salt               string `json:"salt"`
	TokenVersion       int    `json:"tokenVersion"`
	Ban                int    `json:"ban"`
	BaoyueVersion      int    `json:"baoyueVersion"`
	DonateVersion      int    `json:"donateVersion"`
	VipType            int    `json:"vipType"`
	ViptypeVersion     int    `json:"viptypeVersion"`
	AnonimousUser      bool   `json:"anonimousUser"`
	Uninitialized      bool   `json:"uninitialized"`
}

type Profile struct {
	Followed                  bool        `json:"followed"`
	BackgroundUrl             string      `json:"backgroundUrl"`
	DetailDescription         string      `json:"detailDescription"`
	BackgroundImgIdStr        string      `json:"backgroundImgIdStr"`
	AvatarImgIdStr            string      `json:"avatarImgIdStr"`
	UserId                    int64       `json:"userId"`
	UserType                  int         `json:"userType"`
	AccountStatus             int         `json:"accountStatus"`
	VipType                   int         `json:"vipType"`
	Gender                    int         `json:"gender"`
	AvatarImgId               int64       `json:"avatarImgId"`
	Nickname                  string      `json:"nickname"`
	BackgroundImgId           int64       `json:"backgroundImgId"`
	Birthday                  int64       `json:"birthday"`
	City                      int         `json:"city"`
	AvatarUrl                 string      `json:"avatarUrl"`
	DefaultAvatar             bool        `json:"defaultAvatar"`
	Province                  int         `json:"province"`
	Experts                   interface{} `json:"experts"`
	ExpertTags                interface{} `json:"expertTags"`
	Mutual                    bool        `json:"mutual"`
	RemarkName                interface{} `json:"remarkName"`
	AuthStatus                int         `json:"authStatus"`
	DjStatus                  int         `json:"djStatus"`
	Description               string      `json:"description"`
	Signature                 string      `json:"signature"`
	Authority                 int         `json:"authority"`
	AvatarImgIdStr1           string      `json:"avatarImgId_str"`
	Followeds                 int         `json:"followeds"`
	Follows                   int         `json:"follows"`
	EventCount                int         `json:"eventCount"`
	AvatarDetail              interface{} `json:"avatarDetail"`
	PlaylistCount             int         `json:"playlistCount"`
	PlaylistBeSubscribedCount int         `json:"playlistBeSubscribedCount"`
}

// TopPlaylistsResult 热门歌单 /top/playlist
type TopPlaylistsResult struct {
	Result
	Playlists []Playlist `json:"playlists"`
	Total     int        `json:"total"`
	More      bool       `json:"more"`
	Cat       string     `json:"cat"`
}

// User 由 /top/playlist 和 /comment/hot 返回的 User 字段合并而成。
// 冲突的地方以 /top/playlist 为准
type User struct {
	DefaultAvatar       bool        `json:"defaultAvatar"`
	Province            int         `json:"province"`
	AuthStatus          int         `json:"authStatus"`
	Followed            bool        `json:"followed"`
	AvatarUrl           string      `json:"avatarUrl"`
	AccountStatus       int         `json:"accountStatus"`
	Gender              int         `json:"gender"`
	City                int         `json:"city"`
	Birthday            int64       `json:"birthday"`
	UserId              int64       `json:"userId"`
	UserType            int         `json:"userType"`
	Nickname            string      `json:"nickname"`
	Signature           string      `json:"signature"`
	Description         string      `json:"description"`
	DetailDescription   string      `json:"detailDescription"`
	AvatarImgId         int64       `json:"avatarImgId"`
	BackgroundImgId     int64       `json:"backgroundImgId"`
	BackgroundUrl       string      `json:"backgroundUrl"`
	Authority           int         `json:"authority"`
	Mutual              bool        `json:"mutual"`
	ExpertTags          interface{} `json:"expertTags"`
	Experts             interface{} `json:"experts"`
	DjStatus            int         `json:"djStatus"`
	VipType             int         `json:"vipType"`
	RemarkName          interface{} `json:"remarkName"`
	AuthenticationTypes int         `json:"authenticationTypes"`
	AvatarDetail        *struct {
		UserType        int    `json:"userType"`
		IdentityLevel   int    `json:"identityLevel"`
		IdentityIconUrl string `json:"identityIconUrl"`
	} `json:"avatarDetail"`
	Anchor             bool   `json:"anchor"`
	AvatarImgIdStr     string `json:"avatarImgIdStr"`
	BackgroundImgIdStr string `json:"backgroundImgIdStr"`
	AvatarImgIdStr1    string `json:"avatarImgId_str"`

	// comment
	LocationInfo   interface{} `json:"locationInfo"`
	LiveInfo       interface{} `json:"liveInfo"`
	Anonym         int         `json:"anonym"`
	CommonIdentity interface{} `json:"commonIdentity"`
	VipRights      *struct {
		Associator *struct {
			VipCode int  `json:"vipCode"`
			Rights  bool `json:"rights"`
		} `json:"associator"`
		MusicPackage *struct {
			VipCode int  `json:"vipCode"`
			Rights  bool `json:"rights"`
		} `json:"musicPackage"`
		RedVipAnnualCount int `json:"redVipAnnualCount"`
		RedVipLevel       int `json:"redVipLevel"`
	} `json:"vipRights"`
}

// Playlist 由 /top/playlist 和 /playlist/detail 返回的 playlist 字段合并而成。
// 冲突的地方以 /playlist/detail 为准
type Playlist struct {
	// from /top/playlist
	TotalDuration int         `json:"totalDuration"`
	Anonimous     bool        `json:"anonimous"`
	CoverStatus   int         `json:"coverStatus"`
	RecommendInfo interface{} `json:"recommendInfo"`
	Alg           string      `json:"alg"`
	// from /playlist/detail
	Id                    int64       `json:"id"`
	Name                  string      `json:"name"`
	CoverImgId            int64       `json:"coverImgId"`
	CoverImgUrl           string      `json:"coverImgUrl"`
	CoverImgIdStr         string      `json:"coverImgId_str"`
	AdType                int         `json:"adType"`
	UserId                int         `json:"userId"`
	CreateTime            int64       `json:"createTime"`
	Status                int         `json:"status"`
	OpRecommend           bool        `json:"opRecommend"`
	HighQuality           bool        `json:"highQuality"`
	NewImported           bool        `json:"newImported"`
	UpdateTime            int64       `json:"updateTime"`
	TrackCount            int         `json:"trackCount"`
	SpecialType           int         `json:"specialType"`
	Privacy               int         `json:"privacy"`
	TrackUpdateTime       int64       `json:"trackUpdateTime"`
	CommentThreadId       string      `json:"commentThreadId"`
	PlayCount             int         `json:"playCount"`
	TrackNumberUpdateTime int64       `json:"trackNumberUpdateTime"`
	SubscribedCount       int         `json:"subscribedCount"`
	CloudTrackCount       int         `json:"cloudTrackCount"`
	Ordered               bool        `json:"ordered"`
	Description           string      `json:"description"`
	Tags                  []string    `json:"tags"`
	UpdateFrequency       interface{} `json:"updateFrequency"`
	BackgroundCoverId     int         `json:"backgroundCoverId"`
	BackgroundCoverUrl    interface{} `json:"backgroundCoverUrl"`
	TitleImage            int         `json:"titleImage"`
	TitleImageUrl         interface{} `json:"titleImageUrl"`
	EnglishTitle          interface{} `json:"englishTitle"`
	OfficialPlaylistType  interface{} `json:"officialPlaylistType"`
	Subscribers           []User      `json:"subscribers"`
	Subscribed            interface{} `json:"subscribed"`
	Creator               User        `json:"creator"`
	Tracks                []Track     `json:"tracks"`
	VideoIds              interface{} `json:"videoIds"`
	Videos                interface{} `json:"videos"`
	TrackIds              []TrackIDs  `json:"trackIds"`
	ShareCount            int         `json:"shareCount"`
	CommentCount          int         `json:"commentCount"`
	RemixVideo            interface{} `json:"remixVideo"`
	SharedUsers           interface{} `json:"sharedUsers"`
	HistorySharedUsers    interface{} `json:"historySharedUsers"`
}

type TrackIDs struct {
	Id         int         `json:"id"`
	V          int         `json:"v"`
	T          int         `json:"t"`
	At         int64       `json:"at"`
	Alg        interface{} `json:"alg"`
	Uid        int         `json:"uid"`
	RcmdReason string      `json:"rcmdReason"`
}

// Track 由 /playlist/detail 和 /playlist/track/all 返回的 tracks、songs 字段合并而成。
// 冲突的地方以 /playlist/track/all 为准。（事实上，二者完全重合）
type Track struct {
	// from /playlist/detail
	// from /playlist/track/all
	Name                 string        `json:"name"`
	Id                   int           `json:"id"`
	Pst                  int           `json:"pst"`
	T                    int           `json:"t"`
	Ar                   []Artist      `json:"ar"`
	Alia                 []interface{} `json:"alia"`
	Pop                  int           `json:"pop"`
	St                   int           `json:"st"`
	Rt                   *string       `json:"rt"`
	Fee                  int           `json:"fee"`
	V                    int           `json:"v"`
	Crbt                 interface{}   `json:"crbt"`
	Cf                   string        `json:"cf"`
	Al                   Album         `json:"al"`
	Dt                   int           `json:"dt"`
	H                    Quality       `json:"h"` // 高品质 e.g. 320k
	M                    Quality       `json:"m"` // 中品质 e.g. 192k
	L                    Quality       `json:"l"` // 低品质 e.g. 128k
	A                    interface{}   `json:"a"`
	Cd                   string        `json:"cd"`
	No                   int           `json:"no"`
	RtUrl                interface{}   `json:"rtUrl"`
	Ftype                int           `json:"ftype"`
	RtUrls               []interface{} `json:"rtUrls"`
	DjId                 int           `json:"djId"`
	Copyright            int           `json:"copyright"`
	SId                  int           `json:"s_id"`
	Mark                 int64         `json:"mark"`
	OriginCoverType      int           `json:"originCoverType"`
	OriginSongSimpleData interface{}   `json:"originSongSimpleData"`
	TagPicList           interface{}   `json:"tagPicList"`
	ResourceState        bool          `json:"resourceState"`
	Version              int           `json:"version"`
	SongJumpInfo         interface{}   `json:"songJumpInfo"`
	EntertainmentTags    interface{}   `json:"entertainmentTags"`
	Single               int           `json:"single"`
	NoCopyrightRcmd      interface{}   `json:"noCopyrightRcmd"`
	Mv                   int           `json:"mv"`
	Mst                  int           `json:"mst"`
	Cp                   int           `json:"cp"`
	Rtype                int           `json:"rtype"`
	Rurl                 interface{}   `json:"rurl"`
	PublishTime          int64         `json:"publishTime"`
}

// Artist 是 Track 中的艺人信息
type Artist struct {
	Id    int           `json:"id"`
	Name  string        `json:"name"`
	Tns   []interface{} `json:"tns"`
	Alias []interface{} `json:"alias"`
}

// Album 是 Track 中的专辑信息
type Album struct {
	Id     int           `json:"id"`
	Name   string        `json:"name"`
	PicUrl string        `json:"picUrl"`
	Tns    []interface{} `json:"tns"`
	Pic    int64         `json:"pic"`
	PicStr string        `json:"pic_str,omitempty"`
}

// Quality 是 Track 中的品质信息：有 H、M、L（可能 A 也是）
type Quality struct {
	Br   int `json:"br"` // 比特率: e.g. 320000
	Fid  int `json:"fid"`
	Size int `json:"size"` // 文件大小 e.g. 4024990
	Vd   int `json:"vd"`
}

// PlaylistDetailResult 歌单详情 /playlist/detail
type PlaylistDetailResult struct {
	Result
	RelatedVideos   interface{}  `json:"relatedVideos"`
	Playlist        Playlist     `json:"playlist"`
	Urls            interface{}  `json:"urls"`
	Privileges      []Privileges `json:"privileges"`
	SharedPrivilege interface{}  `json:"sharedPrivilege"`
	ResEntrance     interface{}  `json:"resEntrance"`
}

type Privileges struct {
	Id                 int         `json:"id"`
	Fee                int         `json:"fee"`
	Payed              int         `json:"payed"`
	RealPayed          int         `json:"realPayed"`
	St                 int         `json:"st"`
	Pl                 int         `json:"pl"`
	Dl                 int         `json:"dl"`
	Sp                 int         `json:"sp"`
	Cp                 int         `json:"cp"`
	Subp               int         `json:"subp"`
	Cs                 bool        `json:"cs"`
	Maxbr              int         `json:"maxbr"`
	Fl                 int         `json:"fl"`
	Pc                 interface{} `json:"pc"`
	Toast              bool        `json:"toast"`
	Flag               int         `json:"flag"`
	PaidBigBang        bool        `json:"paidBigBang"`
	PreSell            bool        `json:"preSell"`
	PlayMaxbr          int         `json:"playMaxbr"`
	DownloadMaxbr      int         `json:"downloadMaxbr"`
	Rscl               interface{} `json:"rscl"`
	FreeTrialPrivilege struct {
		ResConsumable  bool `json:"resConsumable"`
		UserConsumable bool `json:"userConsumable"`
	} `json:"freeTrialPrivilege"`
	ChargeInfoList []struct {
		Rate          int         `json:"rate"`
		ChargeUrl     interface{} `json:"chargeUrl"`
		ChargeMessage interface{} `json:"chargeMessage"`
		ChargeType    int         `json:"chargeType"`
	} `json:"chargeInfoList"`
}

// PlaylistTracksResult 歌单所有歌曲 /playlist/track/all
type PlaylistTracksResult struct {
	Result
	Songs      []Track      `json:"songs"`
	Privileges []Privileges `json:"privileges"`
}

// HotCommentsResult 歌曲评论结果（似乎其他东西的热论也是这个格式的）
// from /comment/hot?id=33984984&type=0
type HotCommentsResult struct {
	Result
	TopComments []interface{} `json:"topComments"`
	HasMore     bool          `json:"hasMore"`
	HotComments []HotComment  `json:"hotComments"`
	Total       int           `json:"total"`
}

type HotComment struct {
	User      User `json:"user"`
	BeReplied []struct {
		User               User        `json:"user"`
		BeRepliedCommentId int64       `json:"beRepliedCommentId"`
		Content            string      `json:"content"`
		Status             int         `json:"status"`
		ExpressionUrl      interface{} `json:"expressionUrl"`
	} `json:"beReplied"`
	PendantData *struct {
		Id       int    `json:"id"`
		ImageUrl string `json:"imageUrl"`
	} `json:"pendantData"`
	ShowFloorComment    interface{} `json:"showFloorComment"`
	Status              int         `json:"status"`
	CommentId           int64       `json:"commentId"`
	Content             string      `json:"content"`
	ContentResource     interface{} `json:"contentResource"`
	Time                int64       `json:"time"`
	TimeStr             string      `json:"timeStr"`
	NeedDisplayTime     bool        `json:"needDisplayTime"`
	LikedCount          int         `json:"likedCount"`
	ExpressionUrl       interface{} `json:"expressionUrl"`
	CommentLocationType int         `json:"commentLocationType"`
	ParentCommentId     int64       `json:"parentCommentId"`
	Decoration          interface{} `json:"decoration"`
	RepliedMark         interface{} `json:"repliedMark"`
	Liked               bool        `json:"liked"`
}

// LyricResult 歌词结果 /lyric
type LyricResult struct {
	Result
	Sgc       bool      `json:"sgc"`
	Sfy       bool      `json:"sfy"`
	Qfy       bool      `json:"qfy"`
	Lrc       Lyric     `json:"lrc"`    // 默认的歌词：原文
	Klyric    Lyric     `json:"klyric"` // 似乎是音译：不一定有
	Tlyric    Lyric     `json:"tlyric"` // 似乎是翻译：不一定有
	TransUser LyricUser `json:"transUser"`
	LyricUser LyricUser `json:"lyricUser"`
	// 似乎没歌词才有下面几个：
	Uncollected bool        `json:"uncollected"`
	NeedDesc    bool        `json:"needDesc"`
	BriefDesc   interface{} `json:"briefDesc"`
}

type Lyric struct {
	Version int    `json:"version"`
	Lyric   string `json:"lyric"` // 一个样本: "[00:04.050]\n[00:12.570]难以忘记初次见你\n[00:16.860]一双迷人的眼睛\n...\n[03:58.970]爱上你是我情非得已\n[04:03.000]\n"
}

// LyricUser 贡献歌词的用户
type LyricUser struct {
	Id       int    `json:"id"`
	Status   int    `json:"status"`
	Demand   int    `json:"demand"`
	Userid   int    `json:"userid"`
	Nickname string `json:"nickname"`
	Uptime   int64  `json:"uptime"`
}
