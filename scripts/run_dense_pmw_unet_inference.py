#!/usr/bin/env python3
"""Run the GEO+ERA5 deterministic UNet on the dense-PMW timeline."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch,xarray as xr,yaml
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from geo2wf.training import build_model,resolve_runtime_config
from scripts.export_geo_sar_geotiffs import ERA5_CHANNELS,_read_manifest
from scripts.save_deterministic_baseline_fields import _prepare_sample
DATA=ROOT/'inference/inf_data'; DENSE=ROOT/'inference/pmw_pred'; OUTPUT=ROOT/'inference/dense_unet'
CONFIG=ROOT/'configs/config_geo_sar_10bands_era5_residual.yaml'
CHECKPOINT=ROOT/'logs/20260730-132206_config_geo_sar_10bands_era5_residual/checkpoints/epoch=038-step=4758.ckpt'
STATS=ROOT/'data/geotiff/geo_sar_10bands_era5/stats.json'
def args():
 p=argparse.ArgumentParser(); p.add_argument('--storm',default='EP182023'); p.add_argument('--data-root',type=Path,default=DATA); p.add_argument('--dense-root',type=Path,default=DENSE); p.add_argument('--output-root',type=Path,default=OUTPUT); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--checkpoint',type=Path,default=CHECKPOINT); p.add_argument('--stats',type=Path,default=STATS); p.add_argument('--max-geo-gap-minutes',type=float,default=15); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--limit',type=int); p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); return p.parse_args()
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(8*1024*1024): h.update(block)
 return h.hexdigest()
def main():
 a=args(); storm=a.storm.upper()
 for path in (a.config,a.checkpoint,a.stats):
  if not path.is_file(): raise FileNotFoundError(path)
 dense=pd.read_csv(a.dense_root/storm/f'{storm}.csv'); dense['parsed_time']=pd.to_datetime(dense.timestamp,utc=True); dense=dense.sort_values('parsed_time').reset_index(drop=True)
 if a.limit is not None: dense=dense.head(a.limit)
 records=_read_manifest(a.data_root/'index-files/observation_manifest_v6.csv',a.data_root); geos=sorted([r for r in records if r.storm_id==storm and r.source_type=='geo'],key=lambda r:r.timestamp); era5_record=next(r for r in records if r.storm_id==storm and r.source_type=='era5')
 with xr.open_dataset(era5_record.path,group='rectilinear',engine='h5netcdf',decode_times=True) as source: era5=source[list(ERA5_CHANNELS)].load()
 config=resolve_runtime_config(yaml.safe_load(a.config.read_text())); stats=json.loads(a.stats.read_text()); model=build_model(config); checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval().to(a.device)
 prepared=[]; skipped=[]
 for row in tqdm(dense.itertuples(index=False),total=len(dense),desc='Preparing GEO+ERA5'):
  geo=min(geos,key=lambda g:abs((g.timestamp-row.parsed_time).total_seconds())); gap=abs((geo.timestamp-row.parsed_time).total_seconds())/60
  if gap>a.max_geo_gap_minutes: skipped.append({'observation_id':row.observation_id,'timestamp':row.timestamp,'reason':'geo_gap','geo_gap_minutes':gap}); continue
  sample,valid=_prepare_sample(geo,era5,stats); prepared.append((row,geo,sample,valid,gap))
 output_dir=a.output_root/storm/'wind-fields'; output_dir.mkdir(parents=True,exist_ok=True); rows=[]
 with torch.inference_mode():
  for start in tqdm(range(0,len(prepared),a.batch_size),desc='UNet inference'):
   chunk=prepared[start:start+a.batch_size]; batch={key:torch.cat([x[2][key] for x in chunk]).to(a.device) for key in chunk[0][2]}; fields=model.predict_physical(batch).detach().float().cpu().numpy()[:,0]
   for field,(row,geo,_,valid,gap) in zip(fields,chunk):
    filename=str(row.observation_id).replace(':','_')+'.npz'; path=output_dir/filename; np.savez_compressed(path,wind_field_ms=field.astype(np.float32),valid_mask=valid.numpy().astype(np.uint8),timeline_observation_id=np.asarray(row.observation_id),timeline_timestamp=np.asarray(row.timestamp),geo_observation_id=np.asarray(geo.observation_id),geo_timestamp=np.asarray(geo.timestamp.isoformat()))
    rows.append({'storm_id':storm,'observation_id':row.observation_id,'timestamp':row.timestamp,'geo_observation_id':geo.observation_id,'geo_timestamp':geo.timestamp.isoformat(),'geo_gap_minutes':gap,'npz_path':str(path.relative_to(a.output_root)),'array':'wind_field_ms','shape':'x'.join(map(str,field.shape)),'dtype':'float32'})
 a.output_root.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output_root/'dense-unet-fields-manifest.csv',index=False); pd.DataFrame(skipped).to_csv(a.output_root/'dense-unet-skipped.csv',index=False)
 metadata={'schema_version':1,'completed_utc':datetime.now(timezone.utc).isoformat(),'storm':storm,'fields':len(rows),'skipped':len(skipped),'checkpoint':{'path':str(a.checkpoint.resolve()),'sha256':sha256(a.checkpoint)},'config':str(a.config.resolve()),'stats':str(a.stats.resolve()),'timeline_source':str((a.dense_root/storm/f'{storm}.csv').resolve()),'model_inputs':['geostationary','era5'],'dense_pmw_used_as_input':False,'device':a.device}
 (a.output_root/'dense-unet-run-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n'); print(f'Saved {len(rows)} GEO+ERA5 UNet fields; skipped {len(skipped)}')
if __name__=='__main__': main()
