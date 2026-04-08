import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import copernicusmarine
except ModuleNotFoundError:
    copernicusmarine = None
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from config import FileAddress, MapConfig


# ──────────────────────────────────────────────────────────────────────────────
# CMEMS Downloader
# ──────────────────────────────────────────────────────────────────────────────

class CMEMSDataDownloader:

    DATASETS = {
        "horizontal": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        "vertical":   "cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m",
    }

    def __init__(self):
        self.username   = MapConfig.cmems_usrname
        self.password   = MapConfig.cmems_pswd
        self.output_dir = Path(FileAddress.cmems)
        self.centers    = MapConfig.operation_center

        if copernicusmarine is None:
            raise ImportError("copernicusmarine is required only when use_downloader=True. Please install it or set MapConfig.use_downloader=False.")
        if not self.username or not self.password:
            raise ValueError("CMEMS credentials missing in config.py")

    def _bbox(self, center_name: str) -> Tuple:
        lon_c, lat_c, half, radius = self.centers[center_name]
        return (lon_c - half, lat_c - half, lon_c + half, lat_c + half), (lon_c, lat_c), radius

    def download(
        self,
        center_name: str,
        mission_days: int = 2,
        depths: Optional[List[float]] = None,
    ) -> Tuple[str, str, Tuple[float, float], float]:
        depths = depths or [0, 10, 50, 100, 200, 300, 400, 600]
        bbox, origin, op_radius = self._bbox(center_name)
        tag = center_name.replace(" ", "_")
        t0 = datetime.now(timezone.utc)
        t1 = t0 + timedelta(days=mission_days)

        files = {}
        for kind, (variables, suffix) in {
            "horizontal": (["uo", "vo"], "horizontal"),
            "vertical":   (["wo"],       "vertical"),
        }.items():
            fname = self.output_dir / f"currents_{suffix}_{tag}_{t0:%Y%m%d}_{t1:%Y%m%d}.nc"
            print(f"Downloading {kind}: {fname}")
            copernicusmarine.subset(
                dataset_id=self.DATASETS[kind],
                username=self.username,
                password=self.password,
                start_datetime=t0,
                end_datetime=t1,
                minimum_longitude=bbox[0], maximum_longitude=bbox[2],
                minimum_latitude=bbox[1],  maximum_latitude=bbox[3],
                minimum_depth=min(depths), maximum_depth=max(depths),
                variables=variables,
                output_filename=str(fname),
                force_download=True,
            )
            files[kind] = str(fname)
        print(f"\n{'=' * 72}")
        print("Download completed.")
        print(f"{'=' * 72}\n")
        return files["horizontal"], files["vertical"], origin, op_radius

class RealisticCurrentAdapter:
    def __init__(
        self,
        horizontal_path: str,
        vertical_path: Optional[str] = None,
        origin: Optional[Tuple[float, float]] = None,
    ):
        self.origin = origin

        self.enable_temporal = getattr(MapConfig, "use_temporal_current", True)
        self.enable_small_scale_gradient = MapConfig.use_small_scale_gradient
        self.sim_time: Optional[pd.Timestamp] = None
        self.vortices = MapConfig.vortices_vib # 预设一个扰动中心漩涡

        self._load(horizontal_path, vertical_path)

        if origin is not None:
            self.meters_per_deg_lon = MapConfig.meters_per_deg_lat * np.cos(np.radians(origin[1]))

    def _load(self, horiz_path: str, vert_path: Optional[str]):
        ds_h = xr.open_dataset(horiz_path)
        lon_key = "longitude" if "longitude" in ds_h.coords else "lon"
        lat_key = "latitude" if "latitude" in ds_h.coords else "lat"

        self.lon = ds_h[lon_key].values.astype(np.float64)
        self.lat = ds_h[lat_key].values.astype(np.float64)
        self.depth = ds_h["depth"].values.astype(np.float64) if "depth" in ds_h.coords else np.array([0.0])

        u_name = next(n for n in ["uo", "u", "eastward_sea_water_velocity"] if n in ds_h)
        v_name = next(n for n in ["vo", "v", "northward_sea_water_velocity"] if n in ds_h)

        if "time" in ds_h.coords and self.enable_temporal:
            self.time = pd.to_datetime(ds_h["time"].values)
            self.time0 = self.time[0]
            self.time1 = self.time[-1]
            t_hrs_raw = ((self.time - self.time0) / pd.Timedelta(hours=1)).astype(float)
            # 将数据的完整时间跨度压缩到 time_compress_hours 小时内
            compress_hours = float(getattr(MapConfig, "time_compress_hours", 1.0))
            raw_span = float(t_hrs_raw[-1]) if len(t_hrs_raw) > 1 else 1.0
            self._time_scale = compress_hours / raw_span   # 仿真小时 → 压缩坐标
            self.t_hrs = t_hrs_raw / raw_span * compress_hours
            self.sim_time = self.time0
            self.enable_temporal = True
        else:
            self.time = None
            self.time0 = None
            self.time1 = None
            self.t_hrs = None
            self.enable_temporal = False

        u = ds_h[u_name].fillna(0.0)
        v = ds_h[v_name].fillna(0.0)

        if vert_path:
            ds_v = xr.open_dataset(vert_path)
            w_name = next((n for n in ["wo", "w", "vertical_velocity"] if n in ds_v), None)
            if w_name:
                w = ds_v[w_name].fillna(0.0)
                # [核心修复]：处理多源数据的时间分辨率错位
                # 如果开启了时变，且两个数据都带时间戳，则将低频垂直流 w 插值对齐到高频水平流的时间轴上
                if self.enable_temporal and "time" in w.coords and "time" in ds_h.coords:
                    w = w.interp(time=ds_h["time"], kwargs={"fill_value": "extrapolate"})
            else:
                w = xr.zeros_like(u)
        else:
            w = xr.zeros_like(u)

        if self.enable_temporal:
            dims = ("time", "depth", lat_key, lon_key)
            grid = (self.t_hrs, self.depth, self.lat, self.lon)
        else:
            if "time" in u.dims:
                u = u.isel(time=0, drop=True)
            if "time" in v.dims:
                v = v.isel(time=0, drop=True)
            if "time" in w.dims:
                w = w.isel(time=0, drop=True)
            dims = ("depth", lat_key, lon_key)
            grid = (self.depth, self.lat, self.lon)

        u = u.transpose(*dims).values.astype(np.float32)
        v = v.transpose(*dims).values.astype(np.float32)
        w = w.transpose(*dims).values.astype(np.float32)

        kw = dict(bounds_error=False, fill_value=0.0)
        self.u_interp = RegularGridInterpolator(grid, u, **kw)
        self.v_interp = RegularGridInterpolator(grid, v, **kw)
        self.w_interp = RegularGridInterpolator(grid, w, **kw)
        self.max_speed = float(np.sqrt(np.nanmax(u * u + v * v + w * w))) if u.size else 0.0

    def set_simulation_time(self, t: datetime):
        if not self.enable_temporal:
            return None
        ts = pd.Timestamp(t)
        if self.time0 is not None:
            ts = min(max(ts, self.time0), self.time1)
        self.sim_time = ts
        return self.sim_time

    def advance_time(self, dt_seconds: float):
        if not self.enable_temporal:
            return None
        base = self.sim_time if self.sim_time is not None else self.time0
        return self.set_simulation_time(base + pd.Timedelta(seconds=float(dt_seconds)))

    def get_current_at_position(self, pos) -> np.ndarray:
        return self.get_current_at_positions(pos)[0]

    def get_current_at_positions(self, pos, time_value: Optional[datetime] = None) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        lon0, lat0 = self.origin
        lon = lon0 + x / self.meters_per_deg_lon
        lat = lat0 + y / MapConfig.meters_per_deg_lat
        depth = np.clip(z, self.depth.min(), self.depth.max())

        if self.enable_temporal:
            t = pd.Timestamp(time_value or self.sim_time or self.time0)
            t = min(max(t, self.time0), self.time1)
            elapsed_hr = float((t - self.time0) / pd.Timedelta(hours=1))
            # 将仿真经过的实际小时数映射到压缩后的时间坐标
            hr = elapsed_hr * self._time_scale
            hr = np.clip(hr, self.t_hrs.min(), self.t_hrs.max())
            qry = np.column_stack([np.full(len(lon), hr), depth, lat, lon])
        else:
            qry = np.column_stack([depth, lat, lon])

        u = self.u_interp(qry)
        v = self.v_interp(qry)
        w = self.w_interp(qry)
        currents = np.column_stack([v, u, w]).astype(np.float32)

        if self.enable_small_scale_gradient:
            currents = self._small_scale(currents, pos)

        return currents

    def _small_scale(self, base: np.ndarray, pos: np.ndarray) -> np.ndarray:
        # 1. 动态读取漩涡中心坐标和参数
        vtx = self.vortices[0]
        xc, yc, zc = vtx["center"]
        R_vib = vtx["radius"]
        decay_scale = vtx["depth_decay"]
        v_scale = vtx["vertical_scale"]
        
        x, y, z = pos[:, 0] - xc, pos[:, 1] - yc, pos[:, 2]
        
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)

        vs = self.max_speed * v_scale

        mask = (r < R_vib) & (r > 1.0)
        vu = np.zeros_like(x)
        vv = np.zeros_like(x)
        vw = np.zeros_like(x)
        
        # 3. 动态深度衰减
        decay_z = np.exp(-np.abs(z - zc) / decay_scale)

        vu[mask] = -vs * (r[mask] / R_vib) * np.sin(theta[mask]) * decay_z[mask]
        vv[mask] =  vs * (r[mask] / R_vib) * np.cos(theta[mask]) * decay_z[mask]
        vw[mask] = vs * v_scale * np.cos(np.pi * r[mask] / R_vib) * decay_z[mask]
        
        result = base.copy()

        # 5. 生成 X, Y, Z 三个方向的高斯随机扰动 (模拟微尺度海洋湍流)
        noise_level = self.max_speed * 0.05  # 噪声强度为最大流速的 5%
        noise_u = np.random.normal(0, noise_level, size=x.shape)
        noise_v = np.random.normal(0, noise_level, size=x.shape)
        noise_w = np.random.normal(0, noise_level * 0.5, size=x.shape) # Z方向湍流略小

        
        # 6. 叠加所有结构流与随机扰动 
        # (注意：RealisticCurrentAdapter 中 base 的列序是 0:v(北向), 1:u(东向), 2:w(垂向))
        result[:, 0] += vv + noise_v
        result[:, 1] += vu + noise_u
        result[:, 2] += vw + noise_w

        # result[:, 0] += vv
        # result[:, 1] += vu
        # result[:, 2] += vw

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数：供 env.py / 外部脚本调用
# ──────────────────────────────────────────────────────────────────────────────

def build_current(center_name: Optional[str] = None) -> RealisticCurrentAdapter:
    horiz_path = vert_path = None
    origin = op_radius = None
    selected_center = center_name or getattr(MapConfig, "center_name", None) or list(MapConfig.operation_center.keys())[0]

    if MapConfig.use_downloader:
        downloader = CMEMSDataDownloader()
        horiz_path, vert_path, origin, op_radius = downloader.download(center_name=selected_center)
    else:
        horiz_path = str(MapConfig.cmems_horiz_file)
        vert_path  = str(MapConfig.cmems_vert_file)
        lon_c, lat_c, _, radius = MapConfig.operation_center[selected_center]
        origin = (lon_c, lat_c)


    return RealisticCurrentAdapter(
        horizontal_path=horiz_path,
        vertical_path=vert_path,
        origin=origin,
    )


if __name__ == "__main__":
    downloader = CMEMSDataDownloader()
    center_name = "South_China_Sea_Deep_Center"
    downloader.download(center_name = center_name)