# Data

## day_ahead_prices.csv

Day-ahead electricity spot prices for European bidding zones.

**Source:** SMARD (Bundesnetzagentur) — [smard.de/en/downloadcenter/download-market-data](https://www.smard.de/en/downloadcenter/download-market-data/)

**Coverage:** 1 Jan 2016 – 31 Dec 2025 (daily resolution)

**Format:** semicolon-separated (`;`), UTF-8 BOM, 3654 data rows + header

**Note:** The Germany/Luxembourg column (`Germany/Luxembourg [€/MWh] Calculated resolutions`) has no data before October 2018 because SMARD did not report DE/LU as a separate bidding zone for that period. Missing values are represented as `-`.

| Column | Description |
|--------|-------------|
| `Start date` | Start of the day, format `Mon D, YYYY` (e.g. `Jan 1, 2019`) |
| `End date` | End of the day (start date + 1) |
| `Germany/Luxembourg [€/MWh] Calculated resolutions` | DE/LU day-ahead price — the primary column used in this project |
| `∅ DE/LU neighbours [€/MWh] Calculated resolutions` | Average of DE/LU neighbouring zones |
| *(remaining columns)* | Day-ahead prices for Belgium, Denmark 1 & 2, France, Netherlands, Norway 2, Austria, Poland, Sweden 4, Switzerland, Czech Republic, DE/AT/LU, Northern Italy, Slovenia, Hungary |

---

## goerlitz_daily_weather.txt.gz

Daily climate observations for Görlitz weather station (DWD station ID 01684).

**Source:** Deutscher Wetterdienst (DWD) open data — [opendata.dwd.de — daily climate observations, historical](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/)

**Coverage:** 1 Jan 1860 – 31 Dec 2024 (daily resolution)

**Format:** gzip-compressed semicolon-separated (`;`), column names have leading whitespace, 58533 data rows + header. Missing values are represented as `-999`. Polars reads `.gz` files natively with no extra configuration.

The column used in this project is `TMK`. Other available columns:

| Column | Description | Unit |
|--------|-------------|------|
| `STATIONS_ID` | Station identifier (always 1684) | — |
| `MESS_DATUM` | Date in `YYYYMMDD` format | — |
| `QN_3` | Quality level for wind columns | — |
| `FX` | Maximum wind gust | m/s |
| `FM` | Daily mean wind speed | m/s |
| `QN_4` | Quality level for remaining columns | — |
| `RSK` | Daily precipitation | mm |
| `RSKF` | Precipitation type (numeric code) | — |
| `SDK` | Sunshine duration | hours |
| `SHK_TAG` | Snow depth | cm |
| `NM` | Daily mean cloud cover | eighths |
| `VPM` | Daily mean vapour pressure | hPa |
| `PM` | Daily mean air pressure | hPa |
| `TMK` | **Daily mean temperature at 2 m** | °C |
| `UPM` | Daily mean relative humidity | % |
| `TXK` | Daily maximum temperature at 2 m | °C |
| `TNK` | Daily minimum temperature at 2 m | °C |
| `TGK` | Daily minimum temperature at ground level (5 cm) | °C |
| `eor` | End-of-record marker | — |
