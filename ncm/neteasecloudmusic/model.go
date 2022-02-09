package neteasecloudmusic

// Result 似乎是所有接口都会返回有的一个值
// TODO: 也许该给所有结构体加上 Result 的嵌入
type Result struct {
	Code int `json:"code"`
}

const CodeOK = 200

func (r Result) success() bool {
	return r.Code == CodeOK
}

// region API results: 各种 API 请求返回的东西

// TODO: 那些 interface{} 应该被改成实际的结构体，例如 Track

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
}

type Playlist struct {
	Name                  string      `json:"name"`
	Id                    int64       `json:"id"`
	TrackNumberUpdateTime int64       `json:"trackNumberUpdateTime"`
	Status                int         `json:"status"`
	UserId                int64       `json:"userId"`
	CreateTime            int64       `json:"createTime"`
	UpdateTime            int64       `json:"updateTime"`
	SubscribedCount       int         `json:"subscribedCount"`
	TrackCount            int         `json:"trackCount"`
	CloudTrackCount       int         `json:"cloudTrackCount"`
	CoverImgUrl           string      `json:"coverImgUrl"`
	CoverImgId            int64       `json:"coverImgId"`
	Description           string      `json:"description"`
	Tags                  []string    `json:"tags"`
	PlayCount             int         `json:"playCount"`
	TrackUpdateTime       int64       `json:"trackUpdateTime"`
	SpecialType           int         `json:"specialType"`
	TotalDuration         int         `json:"totalDuration"`
	Creator               User        `json:"creator"`
	Tracks                interface{} `json:"tracks"`
	Subscribers           []User      `json:"subscribers"`
	Subscribed            interface{} `json:"subscribed"`
	CommentThreadId       string      `json:"commentThreadId"`
	NewImported           bool        `json:"newImported"`
	AdType                int         `json:"adType"`
	HighQuality           bool        `json:"highQuality"`
	Privacy               int         `json:"privacy"`
	Ordered               bool        `json:"ordered"`
	Anonimous             bool        `json:"anonimous"`
	CoverStatus           int         `json:"coverStatus"`
	RecommendInfo         interface{} `json:"recommendInfo"`
	ShareCount            int         `json:"shareCount"`
	CoverImgIdStr         string      `json:"coverImgId_str"`
	Alg                   string      `json:"alg"`
	CommentCount          int         `json:"commentCount"`
}

type TopPlaylistsResult struct {
	Result
	Playlists []Playlist `json:"playlists"`
	Total     int        `json:"total"`
	More      bool       `json:"more"`
	Cat       string     `json:"cat"`
}

// endregion API results
