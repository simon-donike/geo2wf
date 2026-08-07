const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],NS="http://www.w3.org/2000/svg";
const C={w:600,h:220,l:36,r:8,t:8,b:18};let data,storm,timer,map,layers,geoLayers,sarLayers,pmwLayers,currentMarker,geoFrames=[],sarFrames=[],pmwFrames=[],postProcessing=false,showNwp=false,graphModel="unet_mlp",graphMode="nowcast",forecastModel="convlstm",forecastData=null,forecastRequestId=0,forecastChartState=[],assetBaseUrl="",layerOrderControl;
const NWP_DASHES=["2 2","5 2","8 2","3 2 1 2","10 3","6 2 1 2"];
const DISPLAY_METRICS=["max","rmw"];
const PREDICTION_EDGE_HOURS=6;
const MIN_VALID_RMW_KM=10;
const FORECAST_MATCH_MS=11*60*1000;
const forecastCache=new Map();
const RI_WINDOW_MS=24*3600000,RI_THRESHOLD_MS=30*.514444;
const CATEGORIES=[{label:"C1",value:32.9,color:"#4ca66b"},{label:"C2",value:42.7,color:"#d2b83f"},{label:"C3",value:49.4,color:"#e6943e"},{label:"C4",value:58.1,color:"#db604e"},{label:"C5",value:70.5,color:"#9e4267"}];
const svg=(tag,a={})=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));return n};
const setSvgHidden=(node,hidden)=>node.toggleAttribute("hidden",hidden);
const dt=v=>new Date(v),short=v=>dt(v).toLocaleDateString("en-GB",{day:"numeric",month:"short",timeZone:"UTC"}),full=v=>dt(v).toLocaleString("en-GB",{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZone:"UTC",hour12:false})+" UTC";
const cat=v=>v>=1?`C${v}`:({0:"TS","-1":"TD","-2":"SS","-3":"DB","-4":"EX"}[v]||"—");
const categoryFromWind=v=>v>=137*.514444?5:v>=113*.514444?4:v>=96*.514444?3:v>=83*.514444?2:v>=64*.514444?1:v>=34*.514444?0:-1;
const assetUrl=path=>new URL(path,assetBaseUrl||document.baseURI).href;
const displayStormName=name=>name?name.charAt(0).toUpperCase()+name.slice(1).toLowerCase():"";
const themeColor=token=>getComputedStyle(document.body).getPropertyValue(token).trim();
function chartPointerX(chart,event){
  const chartSvg=chart.querySelector("svg"),matrix=chartSvg?.getScreenCTM?.();
  if(chartSvg?.createSVGPoint&&matrix){const point=chartSvg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;return point.matrixTransform(matrix.inverse()).x}
  const rect=chartSvg.getBoundingClientRect();return(event.clientX-rect.left)/rect.width*C.w
}

const IMAGE_LAYER_META=[{id:"geo",label:"Geostationary imagery",pane:"geoPane"},{id:"pmw",label:"PMW 89–92 GHz",pane:"pmwPane"},{id:"sar",label:"SAR wind fields",pane:"sarPane"}];
function applyImageLayerOrder(){
  const rows=layerOrderControl?.getContainer()?.querySelectorAll(".imagery-layer-row");
  rows?.forEach((row,index)=>{map.getPane(IMAGE_LAYER_META.find(item=>item.id===row.dataset.layer).pane).style.zIndex=350-index*10});
}
function setupLayerOrderControl(control){
  layerOrderControl=control;
  const overlays=control.getContainer().querySelector(".leaflet-control-layers-overlays");
  const labels=[...overlays.querySelectorAll("label")];
  const labelsByLayer=new Map(["geo","sar","pmw"].map((id,index)=>[id,labels[index]]));
  const instruction=document.createElement("div");instruction.className="imagery-layer-instruction";instruction.id="imageryLayerInstruction";instruction.setAttribute("aria-live","polite");instruction.innerHTML="<strong>Imagery stack</strong><span>Drag a grip · top draws above</span>";overlays.prepend(instruction);
  const refreshOrder=()=>{
    [...overlays.querySelectorAll(".imagery-layer-row")].forEach((row,index)=>{row.querySelector(".imagery-layer-rank").textContent=index===0?"TOP":String(index+1)});
    applyImageLayerOrder();
  };
  IMAGE_LAYER_META.forEach(meta=>{
    const label=labelsByLayer.get(meta.id);if(!label)return;
    const row=document.createElement("div");row.className="imagery-layer-row";row.dataset.layer=meta.id;
    const handle=document.createElement("button");handle.type="button";handle.className="imagery-layer-handle";handle.setAttribute("aria-label",`Reorder ${meta.label}. Drag, or use up and down arrow keys.`);handle.setAttribute("aria-describedby",instruction.id);handle.title="Drag to reorder";
    const rank=document.createElement("span");rank.className="imagery-layer-rank";rank.setAttribute("aria-hidden","true");
    row.append(handle,label,rank);overlays.append(row);
    let pointerId=null,startY=0,lastY=0;
    const finishDrag=()=>{
      if(pointerId===null)return;
      const siblings=[...overlays.querySelectorAll(".imagery-layer-row")].filter(item=>item!==row),before=siblings.find(item=>lastY<item.getBoundingClientRect().top+item.offsetHeight/2);
      overlays.insertBefore(row,before||null);pointerId=null;row.style.removeProperty("--drag-offset");row.classList.remove("is-dragging");document.body.classList.remove("is-reordering-imagery");refreshOrder();
    };
    handle.addEventListener("pointerdown",event=>{if(event.button!==0)return;event.preventDefault();pointerId=event.pointerId;startY=lastY=event.clientY;handle.setPointerCapture(pointerId);row.classList.add("is-dragging");document.body.classList.add("is-reordering-imagery")});
    handle.addEventListener("pointermove",event=>{if(event.pointerId!==pointerId)return;event.preventDefault();lastY=event.clientY;row.style.setProperty("--drag-offset",`${lastY-startY}px`)});
    handle.addEventListener("pointerup",finishDrag);handle.addEventListener("pointercancel",finishDrag);handle.addEventListener("lostpointercapture",finishDrag);
    handle.addEventListener("keydown",event=>{if(!["ArrowUp","ArrowDown"].includes(event.key))return;event.preventDefault();const sibling=event.key==="ArrowUp"?row.previousElementSibling:row.nextElementSibling;if(!sibling?.classList.contains("imagery-layer-row"))return;if(event.key==="ArrowUp")overlays.insertBefore(row,sibling);else overlays.insertBefore(sibling,row);refreshOrder();handle.focus()});
  });
  refreshOrder();
}

function initMap(){
  map=L.map("map",{zoomControl:true,preferCanvas:true,attributionControl:false});
  map.createPane("geoPane");map.getPane("geoPane").style.zIndex=350;
  map.createPane("sarPane");map.getPane("sarPane").style.zIndex=340;
  map.createPane("pmwPane");map.getPane("pmwPane").style.zIndex=345;
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",{
    subdomains:"abcd",maxZoom:20,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
  }).addTo(map);
  map.getContainer().insertAdjacentHTML("beforeend",`<span class="map-credit"><a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap</a> · <a href="https://carto.com/attributions" target="_blank" rel="noopener">© CARTO</a></span>`);
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
  const layerControl=L.control.layers(null,{"Geostationary imagery":geoLayers,"SAR wind fields":sarLayers,"PMW 89–92 GHz":pmwLayers},{collapsed:false,position:"topright"}).addTo(map);setupLayerOrderControl(layerControl);
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
function forecastValue(point,source,metric){
  const value=point[source]?.[metric];return Number.isFinite(value)?value:null
}
function forecastPredictionAtTime(metric,time){
  const rows=forecastPoints().map(point=>({time:dt(point.valid_time).getTime(),value:forecastValue(point,"predicted",metric)})).filter(point=>Number.isFinite(point.value));
  if(!rows.length||time<rows[0].time||time>rows[rows.length-1].time)return null;
  const rightIndex=rows.findIndex(point=>point.time>=time),right=rows[rightIndex];
  if(right.time===time||rightIndex===0)return right.value;
  const left=rows[rightIndex-1],fraction=(time-left.time)/(right.time-left.time);return left.value+fraction*(right.value-left.value)
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
    ["predicted","ibtracs"].forEach(source=>{
      const value=forecastValue(point,source,metric);
      if(Number.isFinite(value))values.push(value);
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
function forecastNumber(value,unit){return Number.isFinite(value)?`${value.toFixed(1)} ${unit}`:"Unavailable"}
function signedForecastError(predicted,observed,unit){
  if(!Number.isFinite(predicted)||!Number.isFinite(observed))return"Unavailable";
  const error=predicted-observed;return`${error>=0?"+":""}${error.toFixed(1)} ${unit}`
}
function forecastTooltip(point,metric,def){
  const predicted=forecastValue(point,"predicted",metric),observed=forecastValue(point,"ibtracs",metric),predictedCategory=metric==="max"&&Number.isFinite(predicted)?` · ${cat(categoryFromWind(predicted))}`:"",observedCategory=metric==="max"&&Number.isFinite(observed)?` · ${cat(categoryFromWind(observed))}`:"";
  return`<strong>Issued ${full(point.issue_time)}</strong><br>Valid ${full(point.valid_time)} · +${forecastData.lead_hours} h<br>${forecastData.model?.label||"Forecast"} ${forecastNumber(predicted,def.unit)}${predictedCategory}<br>IBTrACS ${forecastNumber(observed,def.unit)}${observedCategory}<br>Error ${signedForecastError(predicted,observed,def.unit)}<br>Target source ${point.target_source.toUpperCase()}`
}
function forecastChart(metric){
  const points=forecastPoints(),[start,end]=forecastTimeDomain(points),[lo,hi]=forecastDomain(metric,points,start,end),riIntervals=rapidIntensificationIntervals(start,end),def=data.metrics[metric];
  const x=t=>C.l+(dt(t).getTime()-start)/(end-start)*(C.w-C.l-C.r),y=v=>Math.max(C.t,Math.min(C.h-C.b,C.t+(hi-v)/(hi-lo)*(C.h-C.t-C.b)));
  const headingKey=metric==="max"?`<span class="category-key">${CATEGORIES.map(c=>`<b style="--cat:${c.color}">${c.label} ${Math.round(c.value)}</b>`).join("")}</span>`:`<span>${def.note||""} · ${def.unit}</span>`;
  const card=document.createElement("article");card.className="chart-card forecast-chart-card";card.innerHTML=`<div class="chart-heading"><h3>${def.label}</h3>${headingKey}</div><div class="chart"><svg viewBox="0 0 ${C.w} ${C.h}" preserveAspectRatio="xMidYMid meet"></svg></div>`;
  const s=card.querySelector("svg"),defs=svg("defs"),pattern=svg("pattern",{id:`forecast-ri-${metric}`,width:8,height:8,patternUnits:"userSpaceOnUse",patternTransform:"rotate(45)"});pattern.append(svg("line",{class:"ri-hatch-line",x1:0,y1:0,x2:0,y2:8}));defs.append(pattern);s.append(defs);
  riIntervals.forEach(([begin,finish])=>s.append(svg("rect",{class:"ri-region",x:x(begin),y:C.t,width:Math.max(0,x(finish)-x(begin)),height:C.h-C.t-C.b,fill:`url(#forecast-ri-${metric})`})));
  if(metric==="max")CATEGORIES.forEach(c=>{if(c.value>=lo&&c.value<=hi)s.append(svg("line",{class:"category-threshold",x1:C.l,y1:y(c.value),x2:C.w-C.r,y2:y(c.value),stroke:c.color}))});
  [hi,(lo+hi)/2,lo].forEach(v=>{const yp=y(v);s.append(svg("line",{class:"chart-grid",x1:C.l,y1:yp,x2:C.w-C.r,y2:yp}));const text=svg("text",{class:"chart-axis",x:2,y:yp+3});text.textContent=v>=100?Math.round(v):v.toFixed(v<10?1:0);s.append(text)});
  [[short(start),C.l],[short(end),C.w-36]].forEach(([value,xp])=>{const text=svg("text",{class:"chart-axis",x:xp,y:C.h-2});text.textContent=value;s.append(text)});
  const leadBand=svg("rect",{class:"forecast-lead-band",y:C.t,height:C.h-C.t-C.b,hidden:""});s.append(leadBand);
  if(metric==="max")nwpPoints(start,end).forEach((series,index)=>s.append(svg("path",{class:"nwp-path",d:segmentedForecastPath(series.points,point=>point.max,t=>x(t),y,point=>point.time),"stroke-dasharray":NWP_DASHES[index%NWP_DASHES.length],"aria-label":series.label})));
  const forecastPath=segmentedForecastPath(points,point=>forecastValue(point,"predicted",metric),x,y);if(forecastPath)s.append(svg("path",{class:"forecast-path",d:forecastPath}));
  const referencePath=segmentedForecastPath(points,point=>forecastValue(point,"ibtracs",metric),x,y);if(referencePath)s.append(svg("path",{class:"forecast-reference-path",d:referencePath}));
  const issueLine=svg("line",{class:"cursor-line forecast-issue-line","data-start":start,"data-end":end,x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b}),validLine=svg("line",{class:"forecast-valid-line",x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b,hidden:""}),activeSegment=svg("line",{class:"forecast-active-segment",hidden:""}),activeReference=svg("circle",{class:"forecast-active-reference",r:4,hidden:""}),activeForecast=svg("circle",{class:"forecast-active-dot",r:4.5,hidden:""});s.append(issueLine,validLine,activeSegment,activeReference,activeForecast);
  forecastChartState.push({metric,start,end,x,y,leadBand,issueLine,validLine,activeSegment,activeReference,activeForecast});
  card.querySelector(".chart").onpointermove=event=>{
    const pointerX=Math.max(C.l,Math.min(C.w-C.r,chartPointerX(event.currentTarget,event))),fraction=(pointerX-C.l)/(C.w-C.l-C.r),target=start+fraction*(end-start),point=points.reduce((best,item)=>Math.abs(dt(item.issue_time).getTime()-target)<Math.abs(dt(best.issue_time).getTime()-target)?item:best,points[0]);
    if(!point)return;
    const issue=dt(point.issue_time).getTime(),index=storm.records.reduce((best,record,i)=>Math.abs(dt(record.time).getTime()-issue)<Math.abs(dt(storm.records[best].time).getTime()-issue)?i:best,0);$("#timeSlider").value=index;current();updateForecastFocus(point,pointerX);
    const tip=$("#tooltip");tip.innerHTML=forecastTooltip(point,metric,def);tip.style.display="block";tip.style.left=Math.min(event.clientX+12,innerWidth-190)+"px";tip.style.top=event.clientY-70+"px"
  };
  card.querySelector(".chart").onpointerleave=()=>{$("#tooltip").style.display="none";current()};
  return card
}
function nearestForecastIssue(time){
  if(!forecastData?.points?.length)return null;
  const point=forecastData.points.reduce((best,item)=>Math.abs(dt(item.issue_time).getTime()-time)<Math.abs(dt(best.issue_time).getTime()-time)?item:best,forecastData.points[0]);
  return Math.abs(dt(point.issue_time).getTime()-time)<=FORECAST_MATCH_MS?point:null
}
function interpolatedForecastIssue(time){
  if(!forecastData?.points?.length)return null;
  const points=forecastData.points.slice().sort((a,b)=>dt(a.issue_time)-dt(b.issue_time)),rightIndex=points.findIndex(point=>dt(point.issue_time).getTime()>=time);
  if(rightIndex<0||rightIndex===0&&dt(points[0].issue_time).getTime()>time)return null;
  const right=points[rightIndex];if(dt(right.issue_time).getTime()===time)return right;
  const left=points[rightIndex-1],leftTime=dt(left.issue_time).getTime(),rightTime=dt(right.issue_time).getTime(),fraction=(time-leftTime)/(rightTime-leftTime),mix=(source,metric)=>{
    const a=forecastValue(left,source,metric),b=forecastValue(right,source,metric);return Number.isFinite(a)&&Number.isFinite(b)?a+fraction*(b-a):null
  },metrics=new Set([...Object.keys(left.predicted||{}),...Object.keys(right.predicted||{}),...Object.keys(left.ibtracs||{}),...Object.keys(right.ibtracs||{})]),interpolateSource=source=>Object.fromEntries([...metrics].map(metric=>[metric,mix(source,metric)]));
  return{...left,issue_time:new Date(time).toISOString(),valid_time:new Date(time+forecastData.lead_hours*3600000).toISOString(),predicted:interpolateSource("predicted"),ibtracs:interpolateSource("ibtracs"),target_source:"interpolated"}
}
function updateForecastFocus(preferredPoint=null,hoverX=null){
  if(graphMode!=="forecast"||!forecastData||!forecastChartState.length)return;
  const hovered=Boolean(preferredPoint)&&Number.isFinite(hoverX),record=storm.records[+$("#timeSlider").value],issueTime=dt(record.time).getTime(),playing=Boolean(timer),point=preferredPoint||(playing?interpolatedForecastIssue(issueTime):nearestForecastIssue(issueTime));
  if(!point){forecastChartState.forEach(state=>{setSvgHidden(state.validLine,true);setSvgHidden(state.leadBand,true);setSvgHidden(state.activeSegment,true);setSvgHidden(state.activeForecast,true);setSvgHidden(state.activeReference,true)});$("#forecastStatus").hidden=false;$("#forecastStatus").textContent=`No +${forecastData.lead_hours} h forecast for ${full(record.time)}`;return}
  const validTime=dt(point.valid_time).getTime();
  forecastChartState.forEach(state=>{
    const validX=state.x(validTime),issueX=Math.max(C.l,Math.min(C.w-C.r,state.x(dt(point.issue_time).getTime()))),cursorX=hovered?hoverX:issueX,inside=validTime>=state.start&&validTime<=state.end;
    state.issueLine.setAttribute("x1",cursorX);state.issueLine.setAttribute("x2",cursorX);setSvgHidden(state.validLine,!inside);setSvgHidden(state.leadBand,!inside);
    if(inside){state.validLine.setAttribute("x1",validX);state.validLine.setAttribute("x2",validX);state.leadBand.setAttribute("x",Math.min(cursorX,validX));state.leadBand.setAttribute("width",Math.abs(validX-cursorX))}
    [[state.activeForecast,"predicted"],[state.activeReference,"ibtracs"]].forEach(([dot,source])=>{const value=forecastValue(point,source,state.metric),hidden=!inside||!Number.isFinite(value);setSvgHidden(dot,hidden);if(!hidden){dot.setAttribute("cx",validX);dot.setAttribute("cy",state.y(value))}});
    const predicted=forecastValue(point,"predicted",state.metric),predictionAtIssue=forecastPredictionAtTime(state.metric,dt(point.issue_time).getTime()),segmentHidden=!inside||!Number.isFinite(predicted);setSvgHidden(state.activeSegment,segmentHidden);
    if(!segmentHidden){state.activeSegment.setAttribute("x1",cursorX);state.activeSegment.setAttribute("y1",state.y(Number.isFinite(predictionAtIssue)?predictionAtIssue:predicted));state.activeSegment.setAttribute("x2",validX);state.activeSegment.setAttribute("y2",state.y(predicted))}
  });
  $("#forecastStatus").hidden=false;$("#forecastStatus").textContent=`Issued ${full(point.issue_time)} → Valid ${full(point.valid_time)} · +${forecastData.lead_hours} h`
}
function forecastModelsForStorm(selectedStorm=storm){
  const definition=selectedStorm?.forecast;
  if(!definition)return[];
  if(Array.isArray(definition.models))return definition.models;
  return definition.file?[{...definition,id:"convlstm",label:"ConvLSTM",metrics:["max","rmw"]}]:[]
}
function currentForecastMetadata(){
  const models=forecastModelsForStorm();
  return models.find(model=>model.id===forecastModel)||models[0]||null
}
function forecastSupports(metric){
  const metadata=currentForecastMetadata();
  return Boolean(metadata&&(metadata.metrics||["max","rmw"]).includes(metric))
}
function updateForecastModelChrome(){
  const metadata=currentForecastMetadata();if(!metadata)return;
  $("#forecastModelSelector").value=metadata.id;
  $("#forecastLeadBadge").textContent=`+${metadata.lead_hours} h`;
  $("#forecastPredictionLegend").textContent=`${metadata.label} +${metadata.lead_hours} h`;
  $("#forecastMethodNote").textContent=forecastSupports("rmw")?"Deterministic +12 h retrospective validation. Map time is issue time; the highlight is valid time.":`${metadata.label} predicts maximum wind only; RMW is unavailable.`
}
function unavailableForecastChart(metric){
  const def=data.metrics[metric],card=document.createElement("article");card.className="chart-card forecast-metric-unavailable";
  card.innerHTML=`<div class="chart-heading"><h3>${def.label}</h3><span>${def.note||""} · ${def.unit}</span></div><div><strong>Not available for ${forecastData.model?.label||"this model"}</strong><span>Select ConvLSTM to view this metric.</span></div>`;
  return card
}
function forecastCharts(){
  forecastChartState=[];
  if(!forecastData){const card=document.createElement("article"),failed=!$("#forecastNotice").hidden;card.className="chart-card forecast-placeholder";card.innerHTML=failed?"<strong>Forecast data unavailable</strong><span>Use Retry above or return to Nowcast.</span>":"<strong>Loading forecast data…</strong><span>The map and timeline remain available.</span>";$("#charts").replaceChildren(card);return}
  $("#charts").replaceChildren(forecastChart("max"),forecastSupports("rmw")?forecastChart("rmw"):unavailableForecastChart("rmw"))
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
function setAnimationLayerOrder(){applyImageLayerOrder()}
function play(){if(timer)return stop();if($("#animationMode").checked)showAnimationFrame(storm.records[+$("#timeSlider").value]);$("#playButton").textContent="Ⅱ";timer=setInterval(()=>{const s=$("#timeSlider"),next=(+s.value+1)%storm.records.length;if($("#animationMode").checked&&next===0)resetAnimation();s.value=next;current();if($("#animationMode").checked)showAnimationFrame(storm.records[next])},+$("#speedSelector").value)}
function updateModeChrome(){
  const nowcast=graphMode==="nowcast",inferenceAvailable=(storm.available_models||[]).length>0,forecastAvailable=forecastModelsForStorm().length>0;
  $("#nowcastMode").setAttribute("aria-selected",nowcast);$("#forecastMode").setAttribute("aria-selected",!nowcast);$("#nowcastMode").tabIndex=nowcast?0:-1;$("#forecastMode").tabIndex=nowcast?-1:0;$("#forecastMode").disabled=!forecastAvailable;$("#forecastMode").title=forecastAvailable?"Show +12 h forecast graphs":"Forecast data unavailable for this storm";
  $(".model-toolbar").hidden=!nowcast||!inferenceAvailable;$("#forecastToolbar").hidden=nowcast;$("#inferenceNotice").hidden=!nowcast||inferenceAvailable;$("#predictionLegendItem").hidden=!inferenceAvailable;
  $("#nowcastLegend").hidden=!nowcast;$("#forecastLegend").hidden=nowcast;$("#nowcastMethodNote").hidden=!nowcast||!inferenceAvailable;$("#forecastMethodNote").hidden=nowcast;$("#postProcessingControl").hidden=!nowcast;$(".graph-toolbar").hidden=nowcast?!inferenceAvailable:!forecastAvailable;
  if(nowcast){$("#forecastStatus").hidden=true;$("#forecastNotice").hidden=true}else $("#forecastStatus").hidden=$("#forecastNotice").hidden===false;
  $$(".nwp-legend-item").forEach(item=>item.hidden=!showNwp);
  if(forecastAvailable)updateForecastModelChrome()
}
function validForecastPayload(payload,selectedStorm,metadata){
  return payload&&payload.storm_id===selectedStorm.id&&(!payload.model||payload.model.id===metadata.id)&&Number.isFinite(payload.lead_hours)&&Array.isArray(payload.points)&&payload.points.every(point=>point.issue_time&&point.valid_time&&point.predicted&&point.ibtracs)
}
async function loadForecast(force=false){
  const selectedStorm=storm,selectedModel=forecastModel,metadata=currentForecastMetadata();if(!metadata)return;
  const requestId=++forecastRequestId,url=assetUrl(metadata.file);forecastData=null;$("#forecastNotice").hidden=true;$("#forecastStatus").hidden=false;$("#forecastStatus").textContent="Loading forecast data…";forecastCharts();
  if(force)forecastCache.delete(url);
  let pending=forecastCache.get(url);
  if(!pending){pending=fetch(url).then(response=>{if(!response.ok)throw Error(`Could not load forecast data (${response.status})`);return response.json()});forecastCache.set(url,pending)}
  try{
    const payload=await pending;if(requestId!==forecastRequestId||storm!==selectedStorm||forecastModel!==selectedModel||graphMode!=="forecast")return;
    if(!validForecastPayload(payload,selectedStorm,metadata))throw Error("Forecast data has an invalid schema");
    forecastData=payload;$("#forecastNotice").hidden=true;charts();current()
  }catch(error){
    if(forecastCache.get(url)===pending)forecastCache.delete(url);
    if(requestId!==forecastRequestId||storm!==selectedStorm||forecastModel!==selectedModel||graphMode!=="forecast")return;
    forecastData=null;$("#forecastStatus").hidden=true;$("#forecastNoticeText").textContent=error.message;$("#forecastNotice").hidden=false;forecastCharts();console.error(error)
  }
}
function setGraphMode(mode){
  if(mode==="forecast"&&!forecastModelsForStorm().length)return;
  graphMode=mode;updateModeChrome();charts();current();
  if(mode==="forecast"&&!forecastData)loadForecast()
}
function selectStorm(id){
  stop();forecastRequestId++;forecastData=null;storm=data.storms.find(s=>s.id===id);if(graphMode==="forecast"&&!forecastModelsForStorm().length)graphMode="nowcast";switcher();renderMap();
  const availableModels=storm.available_models||[];
  const inferenceAvailable=availableModels.length>0;
  graphModel=availableModels.includes(graphModel)?graphModel:(availableModels.includes("vit")?"vit":availableModels[0]);
  $$("#modelSelector option").forEach(option=>{const definition=data.models[option.value],available=Boolean(definition&&availableModels.includes(option.value));option.hidden=!definition;option.disabled=!available;if(definition)option.textContent=definition.label+(available?"":" (pending)")});
  $("#modelSelector").value=graphModel;$("#modelSelector").disabled=!inferenceAvailable;$("#predictionLegend").textContent=inferenceAvailable?predictionLegendLabel():"";
  postProcessing=inferenceAvailable&&$("#postProcessing").checked;
  const forecastModels=forecastModelsForStorm(),defaultForecast=storm.forecast?.default_model||forecastModels[0]?.id;
  forecastModel=forecastModels.some(model=>model.id===forecastModel)?forecastModel:defaultForecast;
  const forecastSelector=$("#forecastModelSelector");forecastSelector.replaceChildren(...forecastModels.map(model=>{const option=document.createElement("option");option.value=model.id;option.textContent=model.label;return option}));
  forecastSelector.value=forecastModel;forecastSelector.disabled=forecastModels.length<2;
  $("#basinLabel").textContent=`${storm.basin} · ${dt(storm.start).getUTCFullYear()}`;$("#stormHeading").textContent=storm.name?`${displayStormName(storm.name)} Storm track`:"Storm track";$("#stormTitle").textContent=storm.name?`${displayStormName(storm.name)} · ${storm.id}`:storm.id;$("#geoCount").textContent=storm.records.length;$("#sarCount").textContent=storm.sar_matches;$("#pmwCount").textContent=storm.pmw_matches||0;
  const sl=$("#timeSlider");sl.max=storm.records.length-1;sl.value=0;$("#startDate").textContent=short(storm.start);$("#midDate").textContent=short(storm.records[Math.floor(storm.records.length/2)].time);$("#endDate").textContent=short(storm.end);updateModeChrome();charts();current();if(graphMode==="forecast")loadForecast()
}
$("#timeSlider").oninput=()=>{stop();if($("#animationMode").checked)resetAnimation();current();if($("#animationMode").checked)showAnimationFrame(storm.records[+$("#timeSlider").value])};$("#playButton").onclick=play;$("#speedSelector").onchange=()=>{if(timer){stop();play()}};
$("#animationMode").onchange=event=>{setAnimationLayerOrder(event.target.checked);if(event.target.checked){resetAnimation();showAnimationFrame(storm.records[+$("#timeSlider").value])}else restoreManualLayers()};
$("#nowcastMode").onclick=()=>setGraphMode("nowcast");$("#forecastMode").onclick=()=>setGraphMode("forecast");
$(".mode-switch").onkeydown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const mode=event.key==="ArrowRight"?"forecast":"nowcast";if(mode==="forecast"&&$("#forecastMode").disabled)return;setGraphMode(mode);$(mode==="forecast"?"#forecastMode":"#nowcastMode").focus()};
$("#modelSelector").onchange=event=>{graphModel=event.target.value;$("#predictionLegend").textContent=predictionLegendLabel();charts();current()};
$("#forecastModelSelector").onchange=event=>{stop();forecastRequestId++;forecastModel=event.target.value;forecastData=null;updateForecastModelChrome();forecastCharts();loadForecast()};
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
function focusTourRapidIntensification(){
  const intervals=rapidIntensificationIntervals(...chartTimeDomain());
  if(!intervals.length)return null;
  const [begin,finish]=intervals.reduce((best,item)=>item[1]-item[0]>best[1]-best[0]?item:best,intervals[0]);
  const lead=graphMode==="forecast"?(forecastData?.lead_hours||12)*3600000:0,target=finish-lead;
  const index=storm.records.reduce((best,record,i)=>Math.abs(dt(record.time).getTime()-target)<Math.abs(dt(storm.records[best].time).getTime()-target)?i:best,0);
  const slider=$("#timeSlider");slider.value=index;current();
  return{begin,finish,index};
}
function exposeTourControls(){
  window.StormSenseTour={setMode:setGraphMode,focusRapidIntensification:focusTourRapidIntensification};
  window.dispatchEvent(new CustomEvent("stormsense:ready"));
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
loadData().then(d=>{data=d;initMap();storm=data.storms[0];selectStorm(storm.id);$("#loading").classList.add("hidden");exposeTourControls()}).catch(e=>{$("#loading").innerHTML=`<p>Could not load explorer data.<br><small>${e.message}</small></p>`;console.error(e)});
