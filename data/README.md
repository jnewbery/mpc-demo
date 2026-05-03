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

---

## dhn/heatgrids_Sachsen.geojson and dhn/heatgrids_Sachsen.csv

Presumed district heating network (DHN) areas in Saxony (Sachsen), Germany, derived from building heat demand data.

**Source:** RWTH-EBC AixDHN — [github.com/RWTH-EBC/AixDHN](https://github.com/RWTH-EBC/AixDHN)

**Coverage:** 434 network polygons across 251 municipalities in Saxony

**Format:** The GeoJSON file contains polygon geometries (EPSG:3035) plus attributes. The CSV is a geometry-free extract with the scalar attributes only (dropping `geometry`, `centroid`, and `DH_cells`).

| Column | Description | Unit |
|--------|-------------|------|
| `ID` | Unique network identifier | — |
| `GEN` | Municipality name | — |
| `ARS` | Regional administrative key | — |
| `LAN` | State (always `Sachsen`) | — |
| `area` | Area of presumed DHN polygon | km² |
| `DH_demand` | Annual district heating demand (private households) | MWh/year |
| `total_demand_cells` | Total annual heat demand of households in DH cells | MWh/year |
| `total_demand_area` | Total annual heat demand in DHN area | MWh/year |
| `share_DH` | Share of DH demand within the DHN area | % |
| `heat_density_DH` | `DH_demand` divided by `area` | GWh/year/km² |
| `heat_density` | `total_demand_area` divided by `area` | GWh/year/km² |
| `DH_supplied_households` | Number of households supplied by the DHN | — |
