const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],NS="http://www.w3.org/2000/svg";
const C={w:600,h:220,l:36,r:8,t:8,b:18};let data,storm,timer,map,layers,geoLayers,sarLayers,pmwLayers,currentMarker,geoFrames=[],sarFrames=[],pmwFrames=[],postProcessing=false,showNwp=false,graphModel="unet_mlp",graphMode="nowcast",forecastMetric="rmw",forecastData=null,forecastRequestId=0,forecastChartState=[],assetBaseUrl="";
const NWP_DASHES=["2 2","5 2","8 2","3 2 1 2","10 3","6 2 1 2"];
const DISPLAY_METRICS=["max","rmw"];
const PREDICTION_EDGE_HOURS=6;
const MIN_VALID_RMW_KM=10;
const FORECAST_MATCH_MS=11*60*1000;
const FORECAST_METRICS=["rmw","r34","r50","r64"];
const QUADRANTS=["ne","se","sw","nw"];
const forecastCache=new Map();
const RI_WINDOW_MS=24*3600000,RI_THRESHOLD_MS=30*.514444;
const CATEGORIES=[{label:"C1",value:32.9,color:"#4ca66b"},{label:"C2",value:42.7,color:"#d2b83f"},{label:"C3",value:49.4,color:"#e6943e"},{label:"C4",value:58.1,color:"#db604e"},{label:"C5",value:70.5,color:"#9e4267"}];
const svg=(tag,a={})=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));return n};
const dt=v=>new Date(v),short=v=>dt(v).toLocaleDateString("en-GB",{day:"numeric",month:"short",timeZone:"UTC"}),full=v=>dt(v).toLocaleString("en-GB",{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZone:"UTC",hour12:false})+" UTC";
const cat=v=>v>=1?`C${v}`:({0:"TS","-1":"TD","-2":"SS","-3":"DB","-4":"EX"}[v]||"—");
const categoryFromWind=v=>v>=137*.514444?5:v>=113*.514444?4:v>=96*.514444?3:v>=83*.514444?2:v>=64*.514444?1:v>=34*.514444?0:-1;
const assetUrl=path=>new URL(path,assetBaseUrl||document.baseURI).href;
const displayStormName=name=>name?name.charAt(0).toUpperCase()+name.slice(1).toLowerCase():"";
const themeColor=token=>getComputedStyle(document.body).getPropertyValue(token).trim();

function initMap(){
  map=L.map("map",{zoomControl:true,preferCanvas:true,attributionControl:false});
  map.createPane("geoPane");map.getPane("geoPane").style.zIndex=350;
  map.createPane("sarPane");map.getPane("sarPane").style.zIndex=340;
  map.createPane("pmwPane");map.getPane("pmwPane").style.zIndex=345;
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{
    maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  map.getContainer().insertAdjacentHTML("beforeend",`<a class="map-credit" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap</a>`);
  const intensityKey=L.control({position:"bottomright"});
  intensityKey.onAdd=()=>{const el=L.DomUtil.create("div","intensity-key");el.innerHTML=`<strong>SAR wind intensity · ${data.sar_color_scale.unit}</strong><i></i><span><b>${data.sar_color_scale.min}</b><b>${data.sar_color_scale.mid}</b><b>${data.sar_color_scale.max}+</b></span>`;return el};
  intensityKey.addTo(map);
  const geostatKey=L.control({position:"bottomright"});
  geostatKey.onAdd=()=>{const el=L.DomUtil.create("div","geostat-key");el.id="geostatKey";el.innerHTML=`<strong>Geostationary · ${data.geostat_color_scale.channel}</strong><i></i><span><b>${data.geostat_color_scale.min} K</b><b>${data.geostat_color_scale.mid} K</b><b>${data.geostat_color_scale.max} K</b></span>`;return el};
  geostatKey.addTo(map);
  const pmwKey=L.control({position:"bottomright"});
  pmwKey.onAdd=()=>{const el=L.DomUtil.create("div","pmw-key");el.id="pmwKey";el.hidden=true;el.innerHTML=`<strong>PMW brightness temperature · ${data.pmw_color_scale.channel}</strong><i></i><span><b>${data.pmw_color_scale.min} K</b><b>${data.pmw_color_scale.mid} K</b><b>${data.pmw_color_scale.max} K</b></span>`;return el};
  pmwKey.addTo(map);
  layers=L.layerGroup().addTo(map);
  geoLayers=L.layerGroup().addTo(map);
  sarLayers=L.layerGroup().addTo(map);
  pmwLayers=L.layerGroup();
  L.control.layers(null,{"Geostationary imagery":geoLayers,"SAR wind fields":sarLayers,"PMW 89–92 GHz":pmwLayers},{collapsed:false,position:"topright"}).addTo(map);
  map.on("overlayadd",event=>{if(event.layer===geoLayers)$("#geostatKey").hidden=false;if(event.layer===pmwLayers)$("#pmwKey").hidden=false});
  map.on("overlayremove",event=>{if(event.layer===geoLayers)$("#geostatKey").hidden=true;if(event.layer===pmwLayers)$("#pmwKey").hidden=true});
}
function switcher(){
  $("#stormSwitcher").innerHTML=`<label><span>Storm</span><select id="stormSelect">${data.storms.map(s=>`<option value="${s.id}" ${s.id===storm.id?"selected":""}>${s.id}${s.name?` · ${displayStormName(s.name)}`:""} · ${s.basin}</option>`).join("")}</select></label>`;
  $("#stormSelect").onchange=event=>selectStorm(event.target.value);
}
function renderMap(){
  const colors={main:themeColor("--main"),secondary:themeColor("--secondary"),highlight:themeColor("--highlight"),blue:themeColor("--system-blue")};
  layers.clearLayers();
  geoLayers.clearLayers();
  sarLayers.clearLayers();
  pmwLayers.clearLayers();
  geoFrames=[];
  sarFrames=[];
  pmwFrames=[];
  const coords=storm.records.map(r=>[r.lat,r.lon]);
  L.polyline(coords,{color:colors.secondary,weight:8,opacity:.65}).addTo(layers);
  L.polyline(coords,{color:colors.highlight,weight:4,opacity:1}).addTo(layers).bindTooltip(`${storm.id} · Storm track`);
  storm.records.filter(r=>r.geo_overlay).forEach(r=>{
    const o=r.geo_overlay;
    const image=L.imageOverlay(assetUrl(o.image),o.bounds,{opacity:1,interactive:true,className:"geo-overlay",pane:"geoPane"})
      .bindTooltip(`<strong>Geostationary · ${o.channel}</strong><br>${full(r.time)}<br>Fixed GeoStat scale: ${data.geostat_color_scale.min}–${data.geostat_color_scale.max} K`,{className:"geo-tooltip"});
    const show=()=>image.addTo(geoLayers);
    const hide=()=>geoLayers.removeLayer(image);
    geoFrames.push({record:r,show,hide});
    if(!$("#animationMode").checked)show();
  });
  storm.records.filter(r=>r.sar_overlay).forEach(r=>{
    const o=r.sar_overlay;
    const image=L.imageOverlay(assetUrl(o.image),o.bounds,{opacity:.95,interactive:true,className:"sar-overlay",pane:"sarPane"})
      .bindTooltip(`<strong>SAR-derived WF</strong><br>${full(r.time)}<br>Observed range: ${o.min.toFixed(1)}–${o.max.toFixed(1)} m/s<br>Shared color scale: 0–60+ m/s`,{className:"sar-tooltip"});
    const dot=L.circleMarker([r.lat,r.lon],{radius:5,color:colors.main,weight:2,fillColor:colors.secondary,fillOpacity:1})
      .bindTooltip(`SAR match · ${full(r.time)}`);
    const show=()=>{image.addTo(sarLayers);dot.addTo(sarLayers)};
    const hide=()=>{sarLayers.removeLayer(image);sarLayers.removeLayer(dot)};
    sarFrames.push({record:r,show,hide});
    if(!$("#animationMode").checked)show();
  });
  (storm.pmw_observations||[]).forEach(observation=>{
    const o=observation.overlay;
    const image=L.imageOverlay(assetUrl(o.image),o.bounds,{opacity:.92,interactive:true,className:"pmw-overlay",pane:"pmwPane"})
      .bindTooltip(`<strong>PMW · ${o.sensor}</strong><br>${full(observation.time)}<br>${o.channel}<br>Observed range: ${o.min.toFixed(1)}–${o.max.toFixed(1)} K<br>Fixed PMW scale: ${data.pmw_color_scale.min}–${data.pmw_color_scale.max} K`,{className:"pmw-tooltip"});
    const dot=L.circleMarker([observation.lat,observation.lon],{radius:4,color:colors.main,weight:1.5,fillColor:colors.blue,fillOpacity:.9})
      .bindTooltip(`PMW · ${o.sensor}<br>${full(observation.time)}`);
    const show=()=>{image.addTo(pmwLayers);dot.addTo(pmwLayers)};
    const hide=()=>{pmwLayers.removeLayer(image);pmwLayers.removeLayer(dot)};
    pmwFrames.push({time:dt(observation.time).getTime(),show,hide});
    if(!$("#animationMode").checked)show();
  });
  currentMarker=L.circleMarker(coords[0],{radius:8,color:colors.secondary,weight:3,fillColor:colors.highlight,fillOpacity:1}).addTo(layers);
  map.fitBounds(L.latLngBounds(coords).pad(.3),{animate:false,maxZoom:6});
}
function chartTimeDomain(){
  const stormStart=dt(storm.start).getTime(),stormEnd=dt(storm.end).getTime();
  return[stormStart,stormEnd];
}
function interpolatedIbtracsWind(time){
  const rows=storm.records.filter(r=>Number.isFinite(r.ibtracs_msw));
  if(!rows.length||time<dt(rows[0].time).getTime()||time>dt(rows[rows.length-1].time).getTime())return null;
  const after=rows.findIndex(r=>dt(r.time).getTime()>=time);
  if(after<0)return null;
  const right=rows[after],rightTime=dt(right.time).getTime();
  if(rightTime===time||after===0)return right.ibtracs_msw;
  const left=rows[after-1],leftTime=dt(left.time).getTime(),fraction=(time-leftTime)/(rightTime-leftTime);
  return left.ibtracs_msw+fraction*(right.ibtracs_msw-left.ibtracs_msw)
}
function rapidIntensificationIntervals(start,end){
  const first=storm.records.find(record=>Number.isFinite(record.ibtracs_msw));
  if(!first)return[];
  const firstTime=dt(first.time).getTime(),firstWind=first.ibtracs_msw;
  const candidates=storm.records.flatMap((record,index)=>{
    const finish=dt(record.time).getTime(),finishWind=record.ibtracs_msw,baselineTime=finish-RI_WINDOW_MS;
    // Some displayed tracks begin less than 24 hours before RI. Extend their
    // first IBTrACS wind backward only for this calculation so the visible rise
    // can still cross the 30 kt threshold.
    const beginWind=baselineTime<firstTime?firstWind:interpolatedIbtracsWind(baselineTime);
    if(!Number.isFinite(finishWind)||!Number.isFinite(beginWind)||finishWind-beginWind<RI_THRESHOLD_MS)return[];
    const previous=index?dt(storm.records[index-1].time).getTime():finish;
    return[[Math.max(start,previous),Math.min(end,finish)]]
  }).filter(([begin,finish])=>finish>begin);
  return candidates.reduce((merged,interval)=>{
    const previous=merged[merged.length-1];
    if(previous&&interval[0]<=previous[1])previous[1]=Math.max(previous[1],interval[1]);else merged.push(interval);
    return merged
  },[]).map(([begin,finish])=>{
    // A trailing 24-hour gain can remain above 30 kt after weakening starts.
    // Shade only through the strongest IBTrACS point in the detected episode.
    const episode=storm.records.filter(record=>{const time=dt(record.time).getTime();return time>=begin&&time<=finish&&Number.isFinite(record.ibtracs_msw)});
    const peak=episode.reduce((best,record)=>!best||record.ibtracs_msw>best.ibtracs_msw?record:best,null);
    return[begin,peak?dt(peak.time).getTime():finish]
  }).filter(([begin,finish])=>finish>begin)
}

function predictionTimeDomain(){
  const padding=PREDICTION_EDGE_HOURS*3600000;
  return[dt(storm.start).getTime()+padding,dt(storm.end).getTime()-padding];
}
function domain(metric,start,end,predictionStart,predictionEnd){
  const nwp=metric === "max" ? nwpPoints(start,end).flatMap(series=>series.points.map(point=>point.max)) : [];
  const values=[...storm.records.filter(r=>{const time=dt(r.time).getTime();return time>=start&&time<=end}).flatMap(r=>{const time=dt(r.time).getTime();return[time>=predictionStart&&time<=predictionEnd?predictionValue(metric,r):null,r.sar?.[metric],metric==="max"?r.ibtracs_msw:null]}),...nwp].filter(Number.isFinite);
  if(!values.length)return[0,1];
  const min=Math.min(...values),max=Math.max(...values);
  const padding=Math.max((max-min)*.06,metric==="rmw"?1:.5);
  return[0,Math.max(1,max+padding)]
}
function nwpPoints(start,end){return showNwp?(storm.nwp||[]).map(series=>({...series,points:series.points.filter(point=>{const time=dt(point.time).getTime();return time>=start&&time<=end&&Number.isFinite(point.max)})})).filter(series=>series.points.length):[]}
function median(values){const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!sorted.length)return null;const m=Math.floor(sorted.length/2);return sorted.length%2?sorted[m]:(sorted[m-1]+sorted[m])/2}
function predictionSourceModel(metric){return graphModel==="unet_mlp"&&metric!=="max"?"unet":graphModel}
function predictionSourceLabel(metric){return data.models[predictionSourceModel(metric)].label}
function predictionLegendLabel(){return graphModel==="unet_mlp"?"UNet+MLP max · UNet spatial diagnostics":data.models[graphModel].label}
function basePredictionValue(metric,record){
  const raw=graphPrediction(record,metric)?.[metric];
  if(metric!=="rmw"||!Number.isFinite(raw)||raw>=MIN_VALID_RMW_KM)return Number.isFinite(raw)?raw:null;
  const target=dt(record.time).getTime(),valid=storm.records.filter(r=>{const value=graphPrediction(r,metric)?.[metric];return Number.isFinite(value)&&value>=MIN_VALID_RMW_KM});
  const rightIndex=valid.findIndex(r=>dt(r.time).getTime()>target);
  if(rightIndex<=0)return null;
  const left=valid[rightIndex-1],right=valid[rightIndex],leftTime=dt(left.time).getTime(),rightTime=dt(right.time).getTime(),fraction=(target-leftTime)/(rightTime-leftTime);
  return graphPrediction(left,metric)[metric]+fraction*(graphPrediction(right,metric)[metric]-graphPrediction(left,metric)[metric])
}
function graphPrediction(record,metric){return record[`${predictionSourceModel(metric)}_prediction`]}
function smoothedValidValue(metric,record){
  const halfWindow=data.postprocessing.smoothing_hours*3600000/2,target=dt(record.time).getTime();
  return median(storm.records.filter(r=>Math.abs(dt(r.time).getTime()-target)<=halfWindow).map(r=>basePredictionValue(metric,r)));
}
function predictionValue(metric,record){
  const sourceModel=predictionSourceModel(metric);
  if(!data.models[sourceModel].metrics.includes(metric))return null;
  const raw=basePredictionValue(metric,record);
  if(!Number.isFinite(raw))return null;
  if(!postProcessing)return raw;
  return smoothedValidValue(metric,record);
}
function chart(metric,def){
  const [start,end]=chartTimeDomain(),[predictionStart,predictionEnd]=predictionTimeDomain(),[lo,hi]=domain(metric,start,end,predictionStart,predictionEnd),riIntervals=rapidIntensificationIntervals(start,end);
  const x=t=>C.l+(dt(t).getTime()-start)/(end-start)*(C.w-C.l-C.r),y=v=>Math.max(C.t,Math.min(C.h-C.b,C.t+(hi-v)/(hi-lo)*(C.h-C.t-C.b)));
  const inWindow=r=>{const time=dt(r.time).getTime();return time>=start&&time<=end},inPredictionWindow=r=>{const time=dt(r.time).getTime();return time>=predictionStart&&time<=predictionEnd};
  const pred=storm.records.filter(inPredictionWindow).map(r=>({...r,plot:predictionValue(metric,r)})).filter(r=>Number.isFinite(r.plot)),sar=storm.records.filter(r=>inWindow(r)&&Number.isFinite(r.sar?.[metric])),ibtracs=metric==="max"?storm.records.filter(r=>inWindow(r)&&Number.isFinite(r.ibtracs_msw)):[],nwp=metric==="max"?nwpPoints(start,end):[];
  const headingKey=metric==="max"?`<span class="category-key">${CATEGORIES.map(c=>`<b style="--cat:${c.color}">${c.label} ${Math.round(c.value)}</b>`).join("")}</span>`:`<span>${def.note||""} · ${def.unit}</span>`;
  const path=(rows,get)=>rows.map((r,i)=>`${i?"L":"M"}${x(r.time).toFixed(1)},${y(get(r)).toFixed(1)}`).join(" ");
  const card=document.createElement("article");card.className="chart-card";card.innerHTML=`<div class="chart-heading"><h3>${def.label}</h3>${headingKey}</div><div class="chart"><svg viewBox="0 0 ${C.w} ${C.h}" preserveAspectRatio="xMidYMid meet"></svg></div>`;
  const s=card.querySelector("svg");
  const defs=svg("defs"),pattern=svg("pattern",{id:`ri-hatch-${metric}`,width:8,height:8,patternUnits:"userSpaceOnUse",patternTransform:"rotate(45)"});pattern.append(svg("line",{class:"ri-hatch-line",x1:0,y1:0,x2:0,y2:8}));defs.append(pattern);s.append(defs);
  riIntervals.forEach(([begin,finish])=>s.append(svg("rect",{class:"ri-region",x:x(begin),y:C.t,width:Math.max(0,x(finish)-x(begin)),height:C.h-C.t-C.b,fill:`url(#ri-hatch-${metric})`})));
  if(metric==="max")CATEGORIES.forEach(c=>{if(c.value>=lo&&c.value<=hi)s.append(svg("line",{class:"category-threshold",x1:C.l,y1:y(c.value),x2:C.w-C.r,y2:y(c.value),stroke:c.color}))});
  [hi,(lo+hi)/2,lo].forEach(v=>{const yp=y(v);s.append(svg("line",{class:"chart-grid",x1:C.l,y1:yp,x2:C.w-C.r,y2:yp}));const t=svg("text",{class:"chart-axis",x:2,y:yp+3});t.textContent=v>=100?Math.round(v):v.toFixed(v<10?1:0);s.append(t)});
  [[short(start),C.l],[short(end),C.w-36]].forEach(([v,xp])=>{const t=svg("text",{class:"chart-axis",x:xp,y:C.h-2});t.textContent=v;s.append(t)});
  nwp.forEach((series,index)=>s.append(svg("path",{class:"nwp-path",d:path(series.points,r=>r.max),"stroke-dasharray":NWP_DASHES[index%NWP_DASHES.length],"aria-label":series.label})));
  if(pred.length)s.append(svg("path",{class:"geo-path",d:path(pred,r=>r.plot)}));
  if(pred.length===1)s.append(svg("circle",{class:"geo-dot",cx:x(pred[0].time),cy:y(pred[0].plot),r:3.2}));
  if(sar.length)s.append(svg("path",{class:"sar-path",d:path(sar,r=>r.sar[metric])}));
  sar.forEach(r=>s.append(svg("circle",{class:"sar-dot",cx:x(r.time),cy:y(r.sar[metric]),r:3.5})));
  if(ibtracs.length)s.append(svg("path",{class:"ibtracs-path",d:path(ibtracs,r=>r.ibtracs_msw)}));
  s.append(svg("line",{class:"cursor-line","data-start":start,"data-end":end,x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b}));
  card.querySelector(".chart").onpointermove=e=>{const rect=e.currentTarget.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((e.clientX-rect.left)/rect.width*C.w-C.l)/(C.w-C.l-C.r))),target=start+fraction*(end-start),i=storm.records.reduce((best,r,index)=>Math.abs(dt(r.time).getTime()-target)<Math.abs(dt(storm.records[best].time).getTime()-target)?index:best,0);$("#timeSlider").value=i;current();const r=storm.records[i],sv=r.sar?.[metric],pv=inPredictionWindow(r)?predictionValue(metric,r):null,raw=graphPrediction(r,metric)?.[metric],tip=$("#tooltip"),predictionLabel=Number.isFinite(pv)?pv.toFixed(1)+" "+def.unit+(postProcessing?" (smoothed"+(Number.isFinite(raw)?"; raw "+raw.toFixed(1):"")+")":""):"Unavailable",modelCategory=graphModel==="unet_mlp"&&metric==="max"&&Number.isFinite(pv)?categoryFromWind(pv):null,modelCategoryLabel=metric==="max"&&Number.isFinite(modelCategory)?` · ${cat(modelCategory)}`:"",riActive=riIntervals.some(([begin,finish])=>dt(r.time).getTime()>=begin&&dt(r.time).getTime()<=finish),ibtracsLabel=metric==="max"&&Number.isFinite(r.ibtracs_msw)?`<br>IBTrACS ${r.ibtracs_msw.toFixed(1)} m/s · ${cat(r.category)}`:"";tip.innerHTML=`<strong>${full(r.time)}</strong><br>${predictionSourceLabel(metric)} ${predictionLabel}${modelCategoryLabel}<br>SAR-derived WF ${Number.isFinite(sv)?sv.toFixed(1)+" "+def.unit:"No observation"}${ibtracsLabel}${riActive?`<br>IBTrACS rapid intensification · ≥30 kt in 24 h`:""}`;tip.style.display="block";tip.style.left=Math.min(e.clientX+12,innerWidth-150)+"px";tip.style.top=e.clientY-45+"px"};
  card.querySelector(".chart").onpointerleave=()=>$("#tooltip").style.display="none";return card
}
function forecastStats(source,metric){
  const values=QUADRANTS.map(quadrant=>source?.[metric]?.[quadrant]).filter(Number.isFinite);
  if(!values.length)return null;
  return{mean:values.reduce((sum,value)=>sum+value,0)/values.length,min:Math.min(...values),max:Math.max(...values)}
}
function forecastValue(point,source,metric){
  const values=point[source];
  if(metric==="max"||metric==="rmw")return Number.isFinite(values?.[metric])?values[metric]:null;
  return forecastStats(values,metric)?.mean??null
}
function forecastPoints(){
  if(!forecastData?.points)return[];
  const start=dt(storm.start).getTime(),end=dt(storm.end).getTime();
  return forecastData.points.filter(point=>dt(point.valid_time).getTime()>=start&&dt(point.issue_time).getTime()<=end)
}
function forecastTimeDomain(points){
  const start=dt(storm.start).getTime(),stormEnd=dt(storm.end).getTime();
  return[start,Math.max(stormEnd,...points.map(point=>dt(point.valid_time).getTime()))]
}
function forecastDomain(metric,points,start,end){
  const values=[];
  points.forEach(point=>{
    ["predicted","ibtracs","sar"].forEach(source=>{
      const value=forecastValue(point,source,metric);
      if(Number.isFinite(value))values.push(value);
      const stats=metric.startsWith("r")&&metric!=="rmw"?forecastStats(point[source],metric):null;
      if(stats)values.push(stats.min,stats.max)
    })
  });
  if(metric==="max")nwpPoints(start,end).forEach(series=>series.points.forEach(point=>values.push(point.max)));
  if(!values.length)return[0,1];
  const max=Math.max(...values),min=Math.min(...values),padding=Math.max((max-min)*.06,metric==="max"?.5:1);
  return[0,Math.max(1,max+padding)]
}
function segmentedForecastPath(rows,get,x,y,getTime=row=>row.valid_time){
  const segments=[];let segment=[];
  rows.forEach(row=>{const value=get(row);if(Number.isFinite(value))segment.push([x(getTime(row)),y(value)]);else if(segment.length){segments.push(segment);segment=[]}});
  if(segment.length)segments.push(segment);
  return segments.map(points=>points.map(([xp,yp],index)=>`${index?"L":"M"}${xp.toFixed(1)},${yp.toFixed(1)}`).join(" ")).join(" ")
}
function forecastBandPath(rows,metric,x,y){
  const segments=[];let segment=[];
  rows.forEach(row=>{const stats=forecastStats(row.predicted,metric);if(stats)segment.push({row,...stats});else if(segment.length){segments.push(segment);segment=[]}});
  if(segment.length)segments.push(segment);
  return segments.map(points=>{
    const upper=points.map((item,index)=>`${index?"L":"M"}${x(item.row.valid_time).toFixed(1)},${y(item.max).toFixed(1)}`).join(" ");
    const lower=[...points].reverse().map(item=>`L${x(item.row.valid_time).toFixed(1)},${y(item.min).toFixed(1)}`).join(" ");
    return`${upper} ${lower} Z`
  }).join(" ")
}
function forecastNumber(value,unit){return Number.isFinite(value)?`${value.toFixed(1)} ${unit}`:"Unavailable"}
function signedForecastError(predicted,observed,unit){
  if(!Number.isFinite(predicted)||!Number.isFinite(observed))return"Unavailable";
  const error=predicted-observed;return`${error>=0?"+":""}${error.toFixed(1)} ${unit}`
}
function quadrantTooltip(source,metric,label){
  if(metric==="max"||metric==="rmw")return"";
  const values=QUADRANTS.map(quadrant=>`${quadrant.toUpperCase()} ${Number.isFinite(source?.[metric]?.[quadrant])?source[metric][quadrant].toFixed(1):"—"}`).join(" · ");
  return`<br>${label} quadrants ${values} km`
}
function forecastTooltip(point,metric,def){
  const predicted=forecastValue(point,"predicted",metric),observed=forecastValue(point,"ibtracs",metric),sar=forecastValue(point,"sar",metric),predictedCategory=metric==="max"&&Number.isFinite(predicted)?` · ${cat(categoryFromWind(predicted))}`:"",observedCategory=metric==="max"&&Number.isFinite(observed)?` · ${cat(categoryFromWind(observed))}`:"";
  return`<strong>Issued ${full(point.issue_time)}</strong><br>Valid ${full(point.valid_time)} · +${forecastData.lead_hours} h<br>Processor ${forecastNumber(predicted,def.unit)}${predictedCategory}<br>IBTrACS ${forecastNumber(observed,def.unit)}${observedCategory}<br>Error ${signedForecastError(predicted,observed,def.unit)}<br>SAR-derived WF ${forecastNumber(sar,def.unit)}<br>Target source ${point.target_source.toUpperCase()}${quadrantTooltip(point.predicted,metric,"Forecast")}${quadrantTooltip(point.ibtracs,metric,"IBTrACS")}`
}
function forecastChart(metric){
  const points=forecastPoints(),[start,end]=forecastTimeDomain(points),[lo,hi]=forecastDomain(metric,points,start,end),riIntervals=rapidIntensificationIntervals(start,end),def=metric==="max"?data.metrics.max:{label:metric==="rmw"?"Radius of maximum wind":`Radius of ${metric.slice(1)}-knot winds`,unit:"km"};
  const x=t=>C.l+(dt(t).getTime()-start)/(end-start)*(C.w-C.l-C.r),y=v=>Math.max(C.t,Math.min(C.h-C.b,C.t+(hi-v)/(hi-lo)*(C.h-C.t-C.b)));
  const selector=metric==="max"?`<span class="category-key">${CATEGORIES.map(c=>`<b style="--cat:${c.color}">${c.label} ${Math.round(c.value)}</b>`).join("")}</span>`:`<div class="structure-selector" role="tablist" aria-label="Wind-field structure metric">${FORECAST_METRICS.map(value=>`<button type="button" role="tab" tabindex="${value===metric?0:-1}" data-metric="${value}" aria-selected="${value===metric}">${value.toUpperCase()}</button>`).join("")}</div>`;
  const card=document.createElement("article");card.className="chart-card forecast-chart-card";card.innerHTML=`<div class="chart-heading"><h3>${metric==="max"?"Maximum wind · +12 h":"Wind-field structure · +12 h"}</h3>${selector}</div><div class="chart"><svg viewBox="0 0 ${C.w} ${C.h}" preserveAspectRatio="xMidYMid meet"></svg></div>`;
  const s=card.querySelector("svg"),defs=svg("defs"),pattern=svg("pattern",{id:`forecast-ri-${metric}`,width:8,height:8,patternUnits:"userSpaceOnUse",patternTransform:"rotate(45)"});pattern.append(svg("line",{class:"ri-hatch-line",x1:0,y1:0,x2:0,y2:8}));defs.append(pattern);s.append(defs);
  riIntervals.forEach(([begin,finish])=>s.append(svg("rect",{class:"ri-region",x:x(begin),y:C.t,width:Math.max(0,x(finish)-x(begin)),height:C.h-C.t-C.b,fill:`url(#forecast-ri-${metric})`})));
  if(metric==="max")CATEGORIES.forEach(c=>{if(c.value>=lo&&c.value<=hi)s.append(svg("line",{class:"category-threshold",x1:C.l,y1:y(c.value),x2:C.w-C.r,y2:y(c.value),stroke:c.color}))});
  [hi,(lo+hi)/2,lo].forEach(v=>{const yp=y(v);s.append(svg("line",{class:"chart-grid",x1:C.l,y1:yp,x2:C.w-C.r,y2:yp}));const text=svg("text",{class:"chart-axis",x:2,y:yp+3});text.textContent=v>=100?Math.round(v):v.toFixed(v<10?1:0);s.append(text)});
  [[short(start),C.l],[short(end),C.w-36]].forEach(([value,xp])=>{const text=svg("text",{class:"chart-axis",x:xp,y:C.h-2});text.textContent=value;s.append(text)});
  const leadBand=svg("rect",{class:"forecast-lead-band",y:C.t,height:C.h-C.t-C.b,hidden:""});s.append(leadBand);
  if(metric==="max")nwpPoints(start,end).forEach((series,index)=>s.append(svg("path",{class:"nwp-path",d:segmentedForecastPath(series.points,point=>point.max,t=>x(t),y,point=>point.time),"stroke-dasharray":NWP_DASHES[index%NWP_DASHES.length],"aria-label":series.label})));
  if(metric!=="max"&&metric!=="rmw"){const band=forecastBandPath(points,metric,x,y);if(band)s.append(svg("path",{class:"forecast-quadrant-band",d:band}))}
  const forecastPath=segmentedForecastPath(points,point=>forecastValue(point,"predicted",metric),x,y);if(forecastPath)s.append(svg("path",{class:"forecast-path",d:forecastPath}));
  const referencePath=segmentedForecastPath(points,point=>forecastValue(point,"ibtracs",metric),x,y);if(referencePath)s.append(svg("path",{class:"forecast-reference-path",d:referencePath}));
  points.filter(point=>Number.isFinite(forecastValue(point,"sar",metric))).forEach(point=>s.append(svg("circle",{class:"forecast-sar-dot",cx:x(point.valid_time),cy:y(forecastValue(point,"sar",metric)),r:3.2})));
  const issueLine=svg("line",{class:"cursor-line forecast-issue-line","data-start":start,"data-end":end,x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b}),validLine=svg("line",{class:"forecast-valid-line",x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b,hidden:""}),activeReference=svg("circle",{class:"forecast-active-reference",r:4,hidden:""}),activeForecast=svg("circle",{class:"forecast-active-dot",r:4.5,hidden:""});s.append(issueLine,validLine,activeReference,activeForecast);
  forecastChartState.push({metric,start,end,x,y,leadBand,validLine,activeReference,activeForecast});
  card.querySelectorAll(".structure-selector button").forEach(button=>button.onclick=()=>{forecastMetric=button.dataset.metric;charts();current()});
  const structureSelector=card.querySelector(".structure-selector");if(structureSelector)structureSelector.onkeydown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const currentIndex=FORECAST_METRICS.indexOf(forecastMetric),offset=event.key==="ArrowRight"?1:-1;forecastMetric=FORECAST_METRICS[(currentIndex+offset+FORECAST_METRICS.length)%FORECAST_METRICS.length];charts();current();document.querySelector(`.structure-selector button[data-metric="${forecastMetric}"]`)?.focus()};
  card.querySelector(".chart").onpointermove=event=>{
    const rect=event.currentTarget.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((event.clientX-rect.left)/rect.width*C.w-C.l)/(C.w-C.l-C.r))),target=start+fraction*(end-start),point=points.reduce((best,item)=>Math.abs(dt(item.valid_time).getTime()-target)<Math.abs(dt(best.valid_time).getTime()-target)?item:best,points[0]);
    if(!point)return;
    const issue=dt(point.issue_time).getTime(),index=storm.records.reduce((best,record,i)=>Math.abs(dt(record.time).getTime()-issue)<Math.abs(dt(storm.records[best].time).getTime()-issue)?i:best,0);$("#timeSlider").value=index;current();updateForecastFocus(point);
    const tip=$("#tooltip");tip.innerHTML=forecastTooltip(point,metric,def);tip.style.display="block";tip.style.left=Math.min(event.clientX+12,innerWidth-190)+"px";tip.style.top=event.clientY-70+"px"
  };
  card.querySelector(".chart").onpointerleave=()=>$("#tooltip").style.display="none";
  return card
}
function nearestForecastIssue(time){
  if(!forecastData?.points?.length)return null;
  const point=forecastData.points.reduce((best,item)=>Math.abs(dt(item.issue_time).getTime()-time)<Math.abs(dt(best.issue_time).getTime()-time)?item:best,forecastData.points[0]);
  return Math.abs(dt(point.issue_time).getTime()-time)<=FORECAST_MATCH_MS?point:null
}
function updateForecastFocus(preferredPoint=null){
  if(graphMode!=="forecast"||!forecastData||!forecastChartState.length)return;
  const record=storm.records[+$("#timeSlider").value],issueTime=dt(record.time).getTime(),point=preferredPoint||nearestForecastIssue(issueTime);
  if(!point){forecastChartState.forEach(state=>{state.validLine.hidden=true;state.leadBand.hidden=true;state.activeForecast.hidden=true;state.activeReference.hidden=true});$("#forecastStatus").hidden=false;$("#forecastStatus").textContent=`No +${storm.forecast.lead_hours} h forecast for ${full(record.time)}`;return}
  const validTime=dt(point.valid_time).getTime();
  forecastChartState.forEach(state=>{
    const validX=state.x(validTime),issueX=Math.max(C.l,Math.min(C.w-C.r,state.x(dt(point.issue_time).getTime()))),inside=validTime>=state.start&&validTime<=state.end;
    state.validLine.hidden=!inside;state.leadBand.hidden=!inside;
    if(inside){state.validLine.setAttribute("x1",validX);state.validLine.setAttribute("x2",validX);state.leadBand.setAttribute("x",Math.min(issueX,validX));state.leadBand.setAttribute("width",Math.abs(validX-issueX))}
    [[state.activeForecast,"predicted"],[state.activeReference,"ibtracs"]].forEach(([dot,source])=>{const value=forecastValue(point,source,state.metric);dot.hidden=!inside||!Number.isFinite(value);if(!dot.hidden){dot.setAttribute("cx",validX);dot.setAttribute("cy",state.y(value))}})
  });
  $("#forecastStatus").hidden=false;$("#forecastStatus").textContent=`Issued ${full(point.issue_time)} → Valid ${full(point.valid_time)} · +${forecastData.lead_hours} h`
}
function forecastCharts(){
  forecastChartState=[];
  if(!forecastData){const card=document.createElement("article"),failed=!$("#forecastNotice").hidden;card.className="chart-card forecast-placeholder";card.innerHTML=failed?"<strong>Forecast data unavailable</strong><span>Use Retry above or return to Nowcast.</span>":"<strong>Loading forecast data…</strong><span>The map and timeline remain available.</span>";$("#charts").replaceChildren(card);return}
  $("#charts").replaceChildren(forecastChart("max"),forecastChart(forecastMetric))
}
function charts(){
  if(!$("#tooltip")){const t=document.createElement("div");t.id="tooltip";t.className="tooltip";document.body.append(t)}
  if(graphMode==="forecast")forecastCharts();else{$("#charts").replaceChildren(...DISPLAY_METRICS.map(m=>chart(m,data.metrics[m])));forecastChartState=[]}
}
function current(){
  const i=+$("#timeSlider").value,r=storm.records[i];currentMarker.setLatLng([r.lat,r.lon]);
  $("#currentDate").textContent=full(r.time);$("#observationCount").textContent=`Observation ${i+1} of ${storm.records.length}`;
  $("#categoryValue").textContent=cat(r.category);
  const time=dt(r.time).getTime();$$(".cursor-line").forEach(l=>{const start=+l.dataset.start,end=+l.dataset.end,visible=time>=start&&time<=end,xp=C.l+(time-start)/(end-start)*(C.w-C.l-C.r);l.style.display=visible?"":"none";if(visible){l.setAttribute("x1",xp);l.setAttribute("x2",xp)}})
  updateForecastFocus()
}
function stop(){clearInterval(timer);timer=null;$("#playButton").textContent="▶"}
function resetAnimation(){geoFrames.forEach(frame=>frame.hide());sarFrames.forEach(frame=>frame.hide());pmwFrames.forEach(frame=>frame.hide())}
function nearestFrame(frames,record,maxGapHours){const target=dt(record.time).getTime(),frame=frames.reduce((best,item)=>!best||Math.abs(item.time-target)<Math.abs(best.time-target)?item:best,null);return frame&&Math.abs(frame.time-target)<=maxGapHours*3600000?frame:null}
function showAnimationFrame(record){geoFrames.find(frame=>frame.record===record)?.show();sarFrames.find(frame=>frame.record===record)?.show();nearestFrame(pmwFrames,record,3)?.show()}
function restoreManualLayers(){resetAnimation();geoFrames.forEach(frame=>frame.show());sarFrames.forEach(frame=>frame.show());pmwFrames.forEach(frame=>frame.show())}
function setAnimationLayerOrder(enabled){map.getPane("sarPane").style.zIndex=enabled?360:340;map.getPane("pmwPane").style.zIndex=enabled?355:345}
function play(){if(timer)return stop();if($("#animationMode").checked)showAnimationFrame(storm.records[+$("#timeSlider").value]);$("#playButton").textContent="Ⅱ";timer=setInterval(()=>{const s=$("#timeSlider"),next=(+s.value+1)%storm.records.length;if($("#animationMode").checked&&next===0)resetAnimation();s.value=next;current();if($("#animationMode").checked)showAnimationFrame(storm.records[next])},+$("#speedSelector").value)}
function updateModeChrome(){
  const nowcast=graphMode==="nowcast",inferenceAvailable=(storm.available_models||[]).length>0,forecastAvailable=Boolean(storm.forecast);
  $("#nowcastMode").setAttribute("aria-selected",nowcast);$("#forecastMode").setAttribute("aria-selected",!nowcast);$("#nowcastMode").tabIndex=nowcast?0:-1;$("#forecastMode").tabIndex=nowcast?-1:0;$("#forecastMode").disabled=!forecastAvailable;$("#forecastMode").title=forecastAvailable?"Show +12 h forecast graphs":"Forecast data unavailable for this storm";
  $(".model-toolbar").hidden=!nowcast||!inferenceAvailable;$("#forecastToolbar").hidden=nowcast;$("#inferenceNotice").hidden=!nowcast||inferenceAvailable;$("#predictionLegendItem").hidden=!inferenceAvailable;
  $("#nowcastLegend").hidden=!nowcast;$("#forecastLegend").hidden=nowcast;$("#nowcastMethodNote").hidden=!nowcast||!inferenceAvailable;$("#forecastMethodNote").hidden=nowcast;$("#postProcessingControl").hidden=!nowcast;$(".graph-toolbar").hidden=nowcast?!inferenceAvailable:!forecastAvailable;
  if(nowcast){$("#forecastStatus").hidden=true;$("#forecastNotice").hidden=true}else $("#forecastStatus").hidden=$("#forecastNotice").hidden===false;
  $$(".nwp-legend-item").forEach(item=>item.hidden=!showNwp)
}
function validForecastPayload(payload,selectedStorm){
  return payload&&payload.storm_id===selectedStorm.id&&Number.isFinite(payload.lead_hours)&&Array.isArray(payload.points)&&payload.points.every(point=>point.issue_time&&point.valid_time&&point.predicted&&point.ibtracs)
}
async function loadForecast(force=false){
  const selectedStorm=storm,metadata=selectedStorm.forecast;if(!metadata)return;
  const requestId=++forecastRequestId,url=assetUrl(metadata.file);forecastData=null;$("#forecastNotice").hidden=true;$("#forecastStatus").hidden=false;$("#forecastStatus").textContent="Loading forecast data…";forecastCharts();
  if(force)forecastCache.delete(url);
  let pending=forecastCache.get(url);
  if(!pending){pending=fetch(url).then(response=>{if(!response.ok)throw Error(`Could not load forecast data (${response.status})`);return response.json()});forecastCache.set(url,pending)}
  try{
    const payload=await pending;if(requestId!==forecastRequestId||storm!==selectedStorm||graphMode!=="forecast")return;
    if(!validForecastPayload(payload,selectedStorm))throw Error("Forecast data has an invalid schema");
    forecastData=payload;$("#forecastNotice").hidden=true;charts();current()
  }catch(error){
    if(forecastCache.get(url)===pending)forecastCache.delete(url);
    if(requestId!==forecastRequestId||storm!==selectedStorm||graphMode!=="forecast")return;
    forecastData=null;$("#forecastStatus").hidden=true;$("#forecastNoticeText").textContent=error.message;$("#forecastNotice").hidden=false;forecastCharts();console.error(error)
  }
}
function setGraphMode(mode){
  if(mode==="forecast"&&!storm.forecast)return;
  graphMode=mode;updateModeChrome();charts();current();
  if(mode==="forecast"&&!forecastData)loadForecast()
}
function selectStorm(id){
  stop();forecastRequestId++;forecastData=null;storm=data.storms.find(s=>s.id===id);if(graphMode==="forecast"&&!storm.forecast)graphMode="nowcast";switcher();renderMap();
  const availableModels=storm.available_models||[];
  const inferenceAvailable=availableModels.length>0;
  graphModel=availableModels.includes(graphModel)?graphModel:(availableModels.includes("vit")?"vit":availableModels[0]);
  $$("#modelSelector option").forEach(option=>{const definition=data.models[option.value],available=Boolean(definition&&availableModels.includes(option.value));option.hidden=!definition;option.disabled=!available;if(definition)option.textContent=definition.label+(available?"":" (pending)")});
  $("#modelSelector").value=graphModel;$("#modelSelector").disabled=!inferenceAvailable;$("#predictionLegend").textContent=inferenceAvailable?predictionLegendLabel():"";
  postProcessing=inferenceAvailable&&$("#postProcessing").checked;
  $("#basinLabel").textContent=`${storm.basin} · ${dt(storm.start).getUTCFullYear()}`;$("#stormHeading").textContent=storm.name?`${displayStormName(storm.name)} Storm track`:"Storm track";$("#stormTitle").textContent=storm.name?`${displayStormName(storm.name)} · ${storm.id}`:storm.id;$("#geoCount").textContent=storm.records.length;$("#sarCount").textContent=storm.sar_matches;$("#pmwCount").textContent=storm.pmw_matches||0;
  const sl=$("#timeSlider");sl.max=storm.records.length-1;sl.value=0;$("#startDate").textContent=short(storm.start);$("#midDate").textContent=short(storm.records[Math.floor(storm.records.length/2)].time);$("#endDate").textContent=short(storm.end);updateModeChrome();charts();current();if(graphMode==="forecast")loadForecast()
}
$("#timeSlider").oninput=()=>{stop();if($("#animationMode").checked)resetAnimation();current();if($("#animationMode").checked)showAnimationFrame(storm.records[+$("#timeSlider").value])};$("#playButton").onclick=play;$("#speedSelector").onchange=()=>{if(timer){stop();play()}};
$("#animationMode").onchange=event=>{setAnimationLayerOrder(event.target.checked);if(event.target.checked){resetAnimation();showAnimationFrame(storm.records[+$("#timeSlider").value])}else restoreManualLayers()};
$("#nowcastMode").onclick=()=>setGraphMode("nowcast");$("#forecastMode").onclick=()=>setGraphMode("forecast");
$(".mode-switch").onkeydown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const mode=event.key==="ArrowRight"?"forecast":"nowcast";if(mode==="forecast"&&$("#forecastMode").disabled)return;setGraphMode(mode);$(mode==="forecast"?"#forecastMode":"#nowcastMode").focus()};
$("#modelSelector").onchange=event=>{graphModel=event.target.value;$("#predictionLegend").textContent=predictionLegendLabel();charts();current()};
$("#showNwp").onchange=event=>{showNwp=event.target.checked;$$(".nwp-legend-item").forEach(item=>item.hidden=!showNwp);charts();current()};
$("#postProcessing").onchange=event=>{postProcessing=event.target.checked;charts();current()};
$("#retryForecast").onclick=()=>loadForecast(true);
const dialog=$("#aboutDialog");$("#helpButton").onclick=()=>dialog.showModal();$("#closeDialog").onclick=$("#confirmDialog").onclick=()=>dialog.close();dialog.onclick=e=>{if(e.target===dialog)dialog.close()};
async function loadRelease(releaseUrl){
  const pointerResponse=await fetch(releaseUrl,{cache:"no-store"});
  if(!pointerResponse.ok)throw Error(`Could not load release pointer (${pointerResponse.status})`);
  const pointer=await pointerResponse.json();
  if(!pointer.manifest)throw Error("Release pointer has no manifest");
  const manifestUrl=new URL(pointer.manifest,releaseUrl);
  const response=await fetch(manifestUrl);
  if(!response.ok)throw Error(`Could not load storm data (${response.status})`);
  assetBaseUrl=new URL(".",manifestUrl).href;
  return response.json()
}
async function loadData(){
  const configuredUrls=window.GEO2WF_EXPLORER_RELEASE_URLS;
  const releaseUrls=Array.isArray(configuredUrls)
    ? configuredUrls.filter(Boolean)
    : [window.GEO2WF_EXPLORER_RELEASE_URL].filter(Boolean);
  if(!releaseUrls.length){
    const manifestUrl=new URL("storm-data.json",document.baseURI);
    const response=await fetch(manifestUrl);
    if(!response.ok)throw Error(`Could not load storm data (${response.status})`);
    assetBaseUrl=new URL(".",manifestUrl).href;
    return response.json()
  }
  const failures=[];
  for(const releaseUrl of releaseUrls){
    try{return await loadRelease(releaseUrl)}
    catch(error){
      failures.push(`${new URL(releaseUrl).host}: ${error.message}`);
      console.warn(`Explorer data source failed: ${releaseUrl}`,error)
    }
  }
  throw Error(`All explorer data sources failed (${failures.join("; ")})`)
}
loadData().then(d=>{data=d;initMap();storm=data.storms[0];selectStorm(storm.id);$("#loading").classList.add("hidden")}).catch(e=>{$("#loading").innerHTML=`<p>Could not load explorer data.<br><small>${e.message}</small></p>`;console.error(e)});
