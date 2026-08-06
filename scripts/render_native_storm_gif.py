"""Render native GEO/PMW/SAR storm animation with a fixed track map."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from export_geo_sar_geotiffs import PMW_CHANNELS,_make_grid,_load_geo_channels,_load_pmw_channels,_load_sar_channels,_read_manifest,_regrid
from export_geostat_images import GEOSTAT_SCALE_MIN_K as TMIN,GEOSTAT_SCALE_MAX_K as TMAX
S,HH,LH=256,72,24; BLACK=(0,0,0); PAPER=(245,247,250)
PMW_LOW=np.array([45,0,75],np.float32); PMW_MID=np.array([204,71,120],np.float32); PMW_HIGH=np.array([240,249,33],np.float32)
DATA=ROOT/'inference/inf_data'; MAN=DATA/'index-files/observation_manifest_v6.csv'
OUT=ROOT/'docs/assets/images/longest-storm-native.gif'; WORLD=ROOT/'scripts/assets/naturalearth-lowres.geojson'
def args():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=DATA); p.add_argument('--manifest',type=Path,default=MAN); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--world-map',type=Path,default=WORLD); p.add_argument('--storm',default='EP182023'); p.add_argument('--fps',type=float,default=10); p.add_argument('--duration',type=float,default=15); p.add_argument('--geo-crop-size',type=int,default=256); p.add_argument('--sar-max',type=float); p.add_argument('--dense-pmw-root',type=Path); p.add_argument('--dense-pmw-geolocation-root',type=Path,help='Directory containing explicit grid_lat/grid_lon sidecars. Defaults to DENSE_PMW_ROOT/geolocation.'); p.add_argument('--dense-wind-root',type=Path); p.add_argument('--vit-wind-root',type=Path); return p.parse_args()
def font(n):
 for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'):
  if Path(p).exists(): return ImageFont.truetype(p,n)
 return ImageFont.load_default()
def temp_panel(f,m):
 f=np.asarray(f,np.float32).squeeze(); m=np.asarray(m,bool).squeeze()&np.isfinite(f); v=np.nan_to_num(np.clip((f-TMIN)/(TMAX-TMIN),0,1)); rgb=np.array([255,255,255],np.float32)+v[...,None]*(np.array([22,82,180])-np.array([255,255,255])); rgb[~m]=BLACK; return Image.fromarray(rgb.astype(np.uint8)).resize((S,S),Image.Resampling.LANCZOS)
def pmw_panel(f,m):
 f=np.asarray(f,np.float32).squeeze(); m=np.asarray(m,bool).squeeze()&np.isfinite(f); v=np.nan_to_num(np.clip((f-TMIN)/(TMAX-TMIN),0,1)); low=PMW_LOW+v[...,None]*2*(PMW_MID-PMW_LOW); high=PMW_MID+(v[...,None]-.5)*2*(PMW_HIGH-PMW_MID); rgb=np.where((v<=.5)[...,None],low,high).astype(np.uint8); rgb[~m]=BLACK; return Image.fromarray(rgb).resize((S,S),Image.Resampling.LANCZOS)
def sar_panel(f,m,sar_max):
 f=np.asarray(f,np.float32).squeeze(); m=np.asarray(m,bool).squeeze()&np.isfinite(f); weight=gaussian_filter(m.astype(np.float32),1.0); f=np.divide(gaussian_filter(np.where(m,f,0.0),1.0),weight,out=np.zeros_like(f),where=weight>1e-4); m=weight>0.2; v=np.nan_to_num(np.clip(f/sar_max,0,1)); g=np.array([0,150,0]); y=np.array([255,210,0]); r=np.array([255,0,0]); rgb=np.where((v<=.5)[...,None],g+v[...,None]*2*(y-g),y+(v[...,None]-.5)*2*(r-y)).astype(np.uint8); rgb[~m]=BLACK; return Image.fromarray(rgb).resize((S,S),Image.Resampling.LANCZOS)
def center_crop(*arrays,size):
 h,w=np.asarray(arrays[0]).shape[-2:]; size=min(size,h,w); y=(h-size)//2; x=(w-size)//2; return tuple(np.asarray(v)[...,y:y+size,x:x+size] for v in arrays)
def mark_center(panel,grid_lat,grid_lon,center):
 distance=(np.asarray(grid_lat)-center[0])**2+(np.asarray(grid_lon)-center[1])**2; y,x=np.unravel_index(np.nanargmin(distance),distance.shape); h,w=distance.shape; x=x*(panel.width-1)/max(w-1,1); y=y*(panel.height-1)/max(h-1,1); image=panel.copy(); draw=ImageDraw.Draw(image); draw.line((x-5,y-5,x+5,y+5),fill=(255,0,0),width=2); draw.line((x-5,y+5,x+5,y-5),fill=(255,0,0),width=2); return image
def track_center_at(timestamp,geos):
 times=np.array([g.timestamp.value for g in geos],dtype=np.float64); value=float(timestamp.value); return (float(np.interp(value,times,[g.ibtracs_center_lat for g in geos])),float(np.interp(value,times,[g.ibtracs_center_lon for g in geos])))
def dense_pmw_table(root,storm):
 candidates=(root/'index-files/shards'/f'{storm}.csv',root/storm/f'{storm}.csv',root/f'{storm}.csv')
 path=next((candidate for candidate in candidates if candidate.exists()),None)
 if path is None: raise FileNotFoundError(f'No densified PMW manifest found for {storm} under {root}')
 table=pd.read_csv(path); table['parsed_time']=pd.to_datetime(table.timestamp,utc=True); return table.sort_values('parsed_time').reset_index(drop=True)
def dense_pmw_path(row,root,storm):
 path=Path(row.path)
 if path.is_absolute() and path.exists(): return path
 candidates=(root/path,root/storm/path,root/storm/path.name,root/path.name)
 resolved=next((candidate for candidate in candidates if candidate.exists()),None)
 if resolved is None: raise FileNotFoundError(f'Densified PMW tensor not found: {path}')
 return resolved
def _aeqd_crs(center):
 return CRS.from_string(f'+proj=aeqd +lat_0={float(center[0]):.12f} +lon_0={float(center[1]):.12f} +datum=WGS84 +units=m +no_defs')
def recenter_geolocation(lat,lon,source_center,target_center):
 """Rigidly move a curvilinear footprint between storm centres in metres."""
 lat=np.asarray(lat,dtype=np.float64); lon=np.asarray(lon,dtype=np.float64)
 if lat.shape!=lon.shape: raise ValueError(f'Latitude/longitude shape mismatch: {lat.shape} != {lon.shape}')
 result_lat=np.full(lat.shape,np.nan,dtype=np.float64); result_lon=np.full(lon.shape,np.nan,dtype=np.float64); ok=np.isfinite(lat)&np.isfinite(lon)
 if not ok.any(): return result_lat,result_lon
 if np.allclose(source_center,target_center,rtol=0,atol=1e-10): result_lat[ok]=lat[ok]; result_lon[ok]=lon[ok]; return result_lat,result_lon
 east,north=transform_coordinates('EPSG:4326',_aeqd_crs(source_center),lon[ok],lat[ok]); moved_lon,moved_lat=transform_coordinates(_aeqd_crs(target_center),'EPSG:4326',east,north); result_lat[ok]=moved_lat; result_lon[ok]=moved_lon; return result_lat,result_lon
def dense_pmw_geolocation_path(row,root,storm):
 return Path(root)/'geolocation'/storm/f'{Path(row.path).stem}.npz'
def synthetic_pmw_grid(row,records_by_id,geometry_cache=None):
 template_id=str(getattr(row,'target_grid_template_observation_id','')).strip()
 if not template_id: raise ValueError(f'Synthetic PMW row {row.observation_id} has no target_grid_template_observation_id')
 if records_by_id is None or template_id not in records_by_id: raise KeyError(f'PMW template observation is unavailable: {template_id}')
 template=records_by_id[template_id]
 if template.ibtracs_center is None: raise ValueError(f'PMW template {template_id} has no IBTrACS centre')
 key=(template_id,template.sensor)
 cached=geometry_cache.get(key) if geometry_cache is not None else None
 if cached is None:
  channel=PMW_CHANNELS[template.sensor][0]; _,template_lat,template_lon=_load_pmw_channels(template,[channel])[channel]; cached=(template_lat,template_lon,template.ibtracs_center)
  if geometry_cache is not None: geometry_cache[key]=cached
 template_lat,template_lon,source_center=cached; target_center=(float(row.ibtracs_center_lat),float(row.ibtracs_center_lon)); return (*recenter_geolocation(template_lat,template_lon,source_center,target_center),target_center,template_id)
def _load_dense_pmw_sidecar(path,row,shape):
 with np.load(path,allow_pickle=False) as bundle:
  lat=bundle['grid_lat'].astype(np.float64); lon=bundle['grid_lon'].astype(np.float64); stored_id=str(bundle['observation_id'].item()); template_id=str(bundle['template_observation_id'].item()); center=tuple(float(v) for v in bundle['target_center'])
 if stored_id!=str(row.observation_id): raise ValueError(f'PMW geolocation sidecar observation mismatch: {stored_id} != {row.observation_id}')
 if lat.shape!=shape or lon.shape!=shape: raise ValueError(f'PMW geolocation sidecar shape {lat.shape}/{lon.shape} does not match tensor {shape}')
 if not np.isfinite(lat).any() or not np.isfinite(lon).any(): raise ValueError(f'PMW geolocation sidecar has no finite coordinates: {path}')
 return lat,lon,center,template_id
def dense_pmw_field(row,root,storm,records_by_id=None,geometry_cache=None,geolocation_root=None):
 tensor=torch.load(dense_pmw_path(row,root,storm),map_location='cpu',weights_only=False); variables=json.loads(row.variables); channel=next((name for name in variables if name.endswith(('89.0V','91.665V'))),None)
 if channel is None: raise ValueError(f'No supported high-frequency V-polarized PMW channel in {variables}')
 index=variables.index(channel); field=np.asarray(tensor[index],dtype=np.float32); sidecar_root=Path(geolocation_root) if geolocation_root is not None else Path(root)/'geolocation'; sidecar=sidecar_root/storm/f'{Path(row.path).stem}.npz'
 if sidecar.is_file(): lat,lon,center,_=_load_dense_pmw_sidecar(sidecar,row,field.shape)
 else:
  lat,lon,center,_=synthetic_pmw_grid(row,records_by_id,geometry_cache)
  if lat.shape!=field.shape: raise ValueError(f'Synthetic PMW tensor {field.shape} does not match template swath {lat.shape} for {row.observation_id}')
 return field,lat,lon,center
def fixed_map(lat,lon,path):
 ok=np.isfinite(lat)&np.isfinite(lon); lat=np.asarray(lat)[ok]; lon=np.asarray(lon)[ok]; midx=(lon.min()+lon.max())/2; midy=(lat.min()+lat.max())/2; span=max(np.ptp(lon),np.ptp(lat),1)*1.25; west,east,south,north=midx-span/2,midx+span/2,midy-span/2,midy+span/2
 def xy(x,y): return ((float(x)-west)/(east-west)*(S-1),(north-float(y))/(north-south)*(S-1))
 im=Image.new('RGB',(S,S),(198,224,239)); d=ImageDraw.Draw(im); step=10 if span>30 else 5
 for x in np.arange(np.floor(west/step)*step,east,step): d.line([xy(x,south),xy(x,north)],fill=(165,196,213))
 for y in np.arange(np.floor(south/step)*step,north,step): d.line([xy(west,y),xy(east,y)],fill=(165,196,213))
 for feat in json.loads(path.read_text())['features']:
  geom=feat.get('geometry') or {}; cs=geom.get('coordinates',[]); polys=[cs] if geom.get('type')=='Polygon' else cs if geom.get('type')=='MultiPolygon' else []
  for poly in polys:
   for j,ring in enumerate(poly):
    if ring and not(max(x for x,y in ring)<west or min(x for x,y in ring)>east or max(y for x,y in ring)<south or min(y for x,y in ring)>north) and j==0: d.polygon([xy(x,y) for x,y in ring],fill=(220,216,194),outline=(107,116,106))
 track=[xy(x,y) for x,y in zip(lon,lat)]; d.line(track,fill=(115,64,82),width=3); return im,track,xy
def compose(storm,g,meta,panels,base,track,xy,track_i,start,end,footprint):
 w=4*S; im=Image.new('RGB',(w,HH+LH+S),PAPER); d=ImageDraw.Draw(im); val=json.loads(meta).get('usa_sshs') if isinstance(meta,str) else None; cat='unavailable' if val is None else str(int(val)); name='OTIS (EP182023)' if storm=='EP182023' else storm; title=f'{name}  |  IBTrACS category: {cat}  |  {g.timestamp:%Y-%m-%d %H:%M UTC}'; b=d.textbbox((0,0),title,font=font(18)); d.text(((w-b[2]+b[0])/2,7),title,fill=(20,27,38),font=font(18)); a,z,y=78,w-78,43; d.line((a,y,z,y),fill=(117,126,140),width=3); x=a+float((g.timestamp-start)/(end-start))*(z-a); d.ellipse((x-6,y-6,x+6,y+6),fill=(239,111,43),outline=(104,42,20),width=2); d.text((a,52),f'{start:%b %d}',fill=(88,97,110),font=font(11)); t=f'{end:%b %d}'; b=d.textbbox((0,0),t,font=font(11)); d.text((z-b[2]+b[0],52),t,fill=(88,97,110),font=font(11))
 mp=base.convert('RGBA'); overlay=Image.new('RGBA',mp.size,(0,0,0,0)); od=ImageDraw.Draw(overlay); od.polygon([xy(lon,lat) for lat,lon in footprint],fill=(255,255,255,58),outline=(174,31,44,230),width=2); mp=Image.alpha_composite(mp,overlay).convert('RGB'); md=ImageDraw.Draw(mp); md.line(track[:track_i+1],fill=(239,111,43),width=4); x,y=xy(g.ibtracs_center_lon,g.ibtracs_center_lat); md.ellipse((x-6,y-6,x+6,y+6),fill='white',outline=(174,31,44),width=3)
 for i,(label,panel) in enumerate(zip(('Storm track','Geostationary','Predicted PMW' if panels.get('dense_pmw') else 'Passive microwave','Predicted Wind Field' if panels.get('dense_wind') else 'SAR'),(mp,panels['geo'],panels['pmw'],panels['sar']))):
  b=d.textbbox((0,0),label,font=font(15)); d.text((i*S+(S-b[2]+b[0])/2,HH+3),label,fill=(37,45,57),font=font(15)); im.paste(panel,(i*S,HH+LH))
 return im
def main():
 a=args(); n=round(a.fps*a.duration); records=_read_manifest(a.manifest,a.data_root); by_id={r.observation_id:r for r in records}; metadata=pd.read_csv(a.manifest,usecols=['observation_id','metadata_ibtracs']).set_index('observation_id').metadata_ibtracs.to_dict(); by={}
 for r in records:
  if r.source_type=='geo' and r.timestamp is not None: by.setdefault(r.storm_id,[]).append(r)
 for v in by.values(): v.sort(key=lambda x:x.timestamp)
 storm=a.storm.upper() if a.storm else max(by,key=lambda k:(by[k][-1].timestamp-by[k][0].timestamp,len(by[k]))); geos=by[storm]; ids=np.linspace(0,len(geos)-1,n).round().astype(int); selected=[geos[i] for i in ids]; sparse={k:sorted([r for r in records if r.storm_id==storm and r.source_type==k and r.timestamp is not None],key=lambda x:x.timestamp) for k in ('pmw','sar')}; pos={'pmw':0,'sar':0}; raw={'pmw':None,'sar':None}; base,track,xy=fixed_map(np.array([g.ibtracs_center_lat for g in geos]),np.array([g.ibtracs_center_lon for g in geos]),a.world_map); black=Image.new('RGB',(S,S),BLACK); frames=[]
 dense=None; dense_pos=0; dense_geometry_cache={}
 if a.dense_pmw_root is not None:
  dense=dense_pmw_table(a.dense_pmw_root,storm)
 wind=None; wind_pos=0
 if a.dense_wind_root is not None:
  wind=pd.read_csv(a.dense_wind_root/'dense-unet-fields-manifest.csv'); wind=wind[wind.storm_id==storm].copy(); wind['parsed_time']=pd.to_datetime(wind.timestamp,utc=True); wind=wind.sort_values('parsed_time').reset_index(drop=True); wind_values=[]
  for path in wind.npz_path:
   with np.load(a.dense_wind_root/path) as bundle: field=bundle['wind_field_ms']; mask=bundle['valid_mask'].astype(bool); wind_values.append(field[mask & np.isfinite(field)])
  wind_scale=float(np.percentile(np.concatenate(wind_values),99))
 if a.vit_wind_root is not None:
  if wind is not None: raise ValueError('Choose only one of --dense-wind-root and --vit-wind-root')
  vit_root=a.vit_wind_root/storm; wind=pd.read_csv(vit_root/'inference-summary.csv'); wind['parsed_time']=pd.to_datetime(wind.observation_timestamp,utc=True); wind=wind.sort_values('parsed_time').reset_index(drop=True); wind_values=[]
  for path in wind.inference_path:
   bundle=torch.load(vit_root/path,map_location='cpu',weights_only=False); field=np.asarray(bundle['output']).squeeze(); mask=np.asarray(bundle['input_mask']).astype(bool); wind_values.append(field[mask & np.isfinite(field)])
  wind_scale=float(np.percentile(np.concatenate(wind_values),99))
 if a.sar_max is None:
  values=[]
  for item in sparse['sar']:
   field,_,_=_load_sar_channels(item,['wind_speed'])['wind_speed']; values.append(np.asarray(field)[np.isfinite(field)])
  sar_max=float(np.percentile(np.concatenate(values),99)) if values else 60.0
 else: sar_max=a.sar_max
 for track_i,g in tqdm(list(zip(ids,selected)),desc=f'Rendering {storm}'):
  gf,glat,glon=_load_geo_channels(g,['CMI_C15'])['CMI_C15']; gf,glat,glon=center_crop(gf,glat,glon,size=a.geo_crop_size); target_lat,target_lon=glat,glon; panels={'geo':temp_panel(gf,np.isfinite(gf)&np.isfinite(target_lat)&np.isfinite(target_lon))}; center=(g.ibtracs_center_lat,g.ibtracs_center_lon)
  panels['dense_pmw']=dense is not None
  if dense is not None:
   while dense_pos<len(dense) and dense.iloc[dense_pos].parsed_time<=g.timestamp:
    raw['pmw']=dense_pmw_field(dense.iloc[dense_pos],a.dense_pmw_root,storm,records_by_id=by_id,geometry_cache=dense_geometry_cache,geolocation_root=a.dense_pmw_geolocation_root); dense_pos+=1
  panels['dense_wind']=wind is not None
  if wind is not None:
   while wind_pos<len(wind) and wind.iloc[wind_pos].parsed_time<=g.timestamp:
    item=wind.iloc[wind_pos]
    if a.vit_wind_root is not None:
     bundle=torch.load(vit_root/item.inference_path,map_location='cpu',weights_only=False); field=np.asarray(bundle['output']).squeeze().copy(); valid=np.asarray(bundle['input_mask']).astype(bool); field[~valid]=np.nan; source_lat=np.asarray(bundle['grid_lat']); source_lon=np.asarray(bundle['grid_lon'])
    else:
     bundle=np.load(a.dense_wind_root/item.npz_path); field=bundle['wind_field_ms'].copy(); valid=bundle['valid_mask'].astype(bool); field[~valid]=np.nan; bundle.close(); source_geo=by_id[item.geo_observation_id]; source_lat,source_lon=_make_grid(source_geo.center[0],source_geo.center[1],256,0.027); source_lat,source_lon=center_crop(source_lat,source_lon,size=field.shape[-1])
    raw['sar']=(field,source_lat,source_lon,track_center_at(item.parsed_time,geos)); wind_pos+=1
  for k in ('pmw','sar'):
   while (k!='pmw' or dense is None) and (k!='sar' or wind is None) and pos[k]<len(sparse[k]) and sparse[k][pos[k]].timestamp<=g.timestamp:
    r=sparse[k][pos[k]]; pos[k]+=1
    if k=='pmw': ch=PMW_CHANNELS[r.sensor][0]; field,lat,lon=_load_pmw_channels(r,[ch])[ch]
    else: field,lat,lon=_load_sar_channels(r,['wind_speed'])['wind_speed']
    raw[k]=(field,lat,lon,r.ibtracs_center if r.ibtracs_center is not None else track_center_at(r.timestamp,geos))
   if raw[k] is None: panels[k]=black
   else: f,lat,lon,source_center=raw[k]; aligned_lat,aligned_lon=recenter_geolocation(lat,lon,source_center,center); f,m=_regrid(f,aligned_lat,aligned_lon,target_lat,target_lon); panels[k]=pmw_panel(f,m) if k=='pmw' else sar_panel(f,m,wind_scale if wind is not None else sar_max)
  for kind in ('geo','pmw','sar'): panels[kind]=mark_center(panels[kind],target_lat,target_lon,center)
  footprint=[(target_lat[0,0],target_lon[0,0]),(target_lat[0,-1],target_lon[0,-1]),(target_lat[-1,-1],target_lon[-1,-1]),(target_lat[-1,0],target_lon[-1,0])]
  frames.append(compose(storm,g,metadata.get(g.observation_id),panels,base,track,xy,int(track_i),selected[0].timestamp,selected[-1].timestamp,footprint))
 sheet=Image.new('RGB',(1280,int(np.ceil(n/10))*44))
 for i,f in enumerate(frames): sheet.paste(f.resize((128,44),Image.Resampling.BILINEAR),((i%10)*128,(i//10)*44))
 general=sheet.quantize(colors=128,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE); colors=general.getpalette()[:384]; pmw_ramp=[]; wind_ramp=[]; green=np.array([0,150,0]); yellow=np.array([255,210,0]); red=np.array([255,0,0]);
 for value in np.linspace(0,1,64):
  pmw_ramp.extend((PMW_LOW+value*2*(PMW_MID-PMW_LOW) if value<=.5 else PMW_MID+(value-.5)*2*(PMW_HIGH-PMW_MID)).round().astype(int).tolist()); wind_ramp.extend((green+value*2*(yellow-green) if value<=.5 else yellow+(value-.5)*2*(red-yellow)).round().astype(int).tolist())
 pal=Image.new('P',(1,1)); pal.putpalette(colors+pmw_ramp+wind_ramp); q=[f.quantize(palette=pal,dither=Image.Dither.NONE) for f in frames]; a.output.parent.mkdir(parents=True,exist_ok=True); q[0].save(a.output,save_all=True,append_images=q[1:],duration=round(1000/a.fps),loop=0,optimize=True,disposal=1); print(f'Rendered {n} frames for {storm}: {a.output} ({a.output.stat().st_size/1e6:.1f} MB)')
if __name__=='__main__': main()
