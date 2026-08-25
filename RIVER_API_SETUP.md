# 하천 수위 API 연동

## 권장 자료

실시간 운영에는 공공데이터포털에서 `기후에너지환경부 한강홍수통제소_표준수문DB`를 활용 신청한 뒤, 한강홍수통제소 OpenAPI 페이지에서 수위자료 요청 URL과 관측소 코드를 확인하는 방식을 권장합니다.

해당 자료는 관측소별 실시간 수위와 유량 등을 제공합니다. 관측소 목록에서 포항 권역과 형산강 수계에 필요한 지점을 선택하세요.

## .env 설정

```env
RIVER_API_URL=
RIVER_API_KEY=
RIVER_API_KEY_PARAM=serviceKey
RIVER_API_KEY_HEADER=
RIVER_STATION_CODES=
RIVER_REFRESH_SECONDS=300
```

`RIVER_API_URL`은 서비스 페이지에 표시된 전체 요청 URL을 넣습니다. 다음 자리표시자를 사용할 수 있습니다.

```text
{service_key}   인증키
{station_code}  관측소 코드
{start_date}    7일 전 날짜 YYYYMMDD
{end_date}      오늘 날짜 YYYYMMDD
{today}         오늘 날짜 YYYYMMDD
```

예시:

```env
RIVER_API_URL=https://기관주소/수위조회?station={station_code}&from={start_date}&to={end_date}&format=json
RIVER_API_KEY=발급받은키
RIVER_API_KEY_PARAM=serviceKey
RIVER_STATION_CODES=관측소코드1,관측소코드2
RIVER_REFRESH_SECONDS=300
```

URL 안에 인증키 자리가 있는 서비스는 다음처럼 사용할 수 있습니다.

```env
RIVER_API_URL=https://기관주소/수위조회?key={service_key}&station={station_code}
RIVER_API_KEY=발급받은키
RIVER_API_KEY_PARAM=
```

인증키를 HTTP 헤더로 전달하는 서비스는 다음처럼 설정합니다.

```env
RIVER_API_URL=https://기관주소/수위조회
RIVER_API_KEY=발급받은키
RIVER_API_KEY_HEADER=Authorization
RIVER_API_KEY_PARAM=
```

Bearer 접두사가 필요한 경우 현재 코드에는 키 값 전체가 헤더 값으로 들어가므로 다음처럼 입력합니다.

```env
RIVER_API_KEY=Bearer 발급받은키
```

## 인식하는 필드

다음과 같은 공공 수문 API 필드를 자동 인식합니다.

```text
관측소 코드: id, station_id, stationCode, wlobscd, wlobsCd
관측소 이름: name, station_name, stationName, wlobsnm
수위: level_m, water_level_m, waterLevel, level, wl, wal
유량: flow_cms, flow, flux, discharge
관측 시각: observed_at, obsTime, tm, datetime, obsrdate
```

JSON과 XML 응답을 모두 처리합니다.

## 확인

서버 실행 후 PowerShell:

```powershell
Invoke-RestMethod `
  "http://127.0.0.1:8000/api/river-levels" |
  ConvertTo-Json -Depth 10
```

정상 예시:

```json
{
  "source": "api",
  "sensors": [
    {
      "id": "관측소코드",
      "name": "관측소명",
      "level_m": 1.23,
      "flow_cms": 15.8,
      "observed_at": "202608061320"
    }
  ]
}
```

`RIVER_API_URL`이 비어 있으면 형산강, 냉천, 곡강천 데모값이 표시됩니다.
