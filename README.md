# NMEA 2000 Wind Data Simulator

실제 요트 레이스 GPS 트랙 데이터와 ERA5 기상 재분석 데이터를 결합하여
NMEA 2000 풍속/풍향 데이터를 시뮬레이션합니다.
ESP32 + iOS BLE 앱 개발을 위한 테스트 데이터 생성이 최종 목적입니다.

## 데이터 소스
- **GPS Track**: Rolex China Sea Race 2026 (홍콩 → 수빅만, 74.3시간)
- **Wind**: ERA5 Reanalysis (ECMWF) + Bavaria 56 Polar 기반 Kalman 보정

## NMEA 2000 PGN Mapping

| PGN | Name | Fields | Update Rate | 시뮬레이터 데이터 |
|-----|------|--------|-------------|-----------------|
| 130306 | Wind Data | AWS, AWA, wind reference | 100ms | `AWS`, `AWA` (Apparent) |
| 130306 | Wind Data | TWS, TWD, wind reference | 100ms | `TWS`, `TWD` (True, ground ref) |
| 129026 | COG & SOG, Rapid Update | COG, SOG | 250ms | `COG`, `SOG` |
| 128259 | Speed, Water Referenced | BSP, water speed type | 1s | `BSP` (polar estimated) |
| 129029 | GNSS Position Data | lat, lon, altitude, time | 1s | `lat`, `lon` |
| 129025 | Position, Rapid Update | lat, lon | 100ms | `lat`, `lon` |
| 127250 | Vessel Heading | heading, deviation, variation | 100ms | `COG` (proxy) |
| 130310 | Environmental Parameters | water temp, air temp, pressure | 500ms | — (미구현) |

### PGN 130306 — Wind Data (핵심)
NMEA 2000에서 풍속/풍향은 **하나의 PGN(130306)**으로 전송되며, `Wind Reference` 필드로 구분:
- `0` = True (ground referenced) → TWS, TWD
- `1` = Magnetic → 미사용
- `2` = Apparent → AWS, AWA
- `3` = True (boat referenced) → TWS, TWA

시뮬레이터는 Apparent(2)를 기본 출력하며, True(0, 3)도 함께 생성합니다.

### PGN 데이터 포맷
```
PGN 130306 (8 bytes):
  Byte 0:     SID (Sequence ID)
  Byte 1-2:   Wind Speed (0.01 m/s resolution)
  Byte 3-4:   Wind Angle (0.0001 rad resolution)
  Byte 5:     Wind Reference (0=True ground, 2=Apparent, 3=True boat)
  Byte 6-7:   Reserved
```

## Quality Flags
| Value | Meaning | Description |
|-------|---------|-------------|
| 0 | Missing | 센서 데이터 누락 (NULL) |
| 1 | Normal | 정상 데이터 |
| 2 | Interpolated | 보간된 데이터 |
| 3 | Suspect | 물리적으로 의심스러운 값 (스파이크) |

## Pipeline

```
GPX Parse → ERA5 Download → Polar Correction → Apparent Wind → Noise Injection → Output
   ↓            ↓                ↓                  ↓               ↓            ↓
 SOG/COG    TWS/TWD(raw)   TWS/TWD(corrected)    AWS/AWA      gaps/spikes    CSV/JSON
```

## 실행

```bash
pip install -r requirements.txt

# 전체 파이프라인
python3 main.py --step all

# 개별 스텝
python3 main.py --step parse      # GPX → SOG/COG
python3 main.py --step era5       # ERA5 다운로드 + 보간
python3 main.py --step correct    # Polar inversion + Kalman 보정
python3 main.py --step calc       # AWS/AWA 계산
python3 main.py --step noise      # 노이즈 + 갭 주입
python3 main.py --step output     # CSV 출력
python3 main.py --step validate   # 시각화 검증

# ESP32 스트리밍
python3 main.py --stream --speed 10
python3 main.py --stream --speed 10 --port /dev/tty.usbserial
```

## 출력 파일
- `data/output/simulated_wind.csv` — 전체 시뮬레이션 데이터
- `data/output/events.csv` — 주입된 이벤트 로그 (gusts, gaps, spikes)
- `data/output/track_data.json` — 대시보드용 JSON
- `data/output/viewer.html` — 인터랙티브 대시보드
- `data/output/validation/` — 검증 플롯 (route map, timeseries, wind rose, gaps)

## 요구사항
- Python 3.11+
- ERA5 API: `~/.cdsapirc`에 CDS Personal Access Token 설정 필요
- [CDS 라이선스 수락](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences) 필요

## 디렉토리 구조
```
├── data/
│   ├── raw/RCSR2026.gpx        # GPS 트랙
│   ├── era5/                    # ERA5 캐시 (netCDF)
│   └── output/                  # 생성된 데이터
├── src/
│   ├── parser.py                # GPX 파싱 + SOG/COG
│   ├── era5.py                  # ERA5 쿼리 + 시공간 보간
│   ├── polar.py                 # Bavaria 56 polar + Kalman 보정
│   ├── wind_calc.py             # AWS/AWA 벡터 계산
│   ├── noise.py                 # 노이즈 + 갭/스파이크 주입
│   ├── output.py                # CSV + Serial 스트리밍
│   └── visualize.py             # 검증 시각화
├── main.py                      # 파이프라인 실행
├── config.yaml                  # 파라미터 설정
└── requirements.txt
```
