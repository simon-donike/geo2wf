const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],NS="http://www.w3.org/2000/svg";
const C={w:600,h:112,l:36,r:8,t:8,b:18};let data,storm,timer,map,layers,geoLayers,sarLayers,currentMarker,geoFrames=[],sarFrames=[],postProcessing=true,showNwp=false,graphModel="model_a",assetBaseUrl="";
const NWP_DASHES=["2 2","5 2","8 2","3 2 1 2","10 3","6 2 1 2"];
const DISPLAY_METRICS=["max","p90","core_mean","rmw"];
const CATEGORIES=[{label:"C1",value:32.9,color:"#4ca66b"},{label:"C2",value:42.7,color:"#d2b83f"},{label:"C3",value:49.4,color:"#e6943e"},{label:"C4",value:58.1,color:"#db604e"},{label:"C5",value:70.5,color:"#9e4267"}];
const svg=(tag,a={})=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));return n};
const dt=v=>new Date(v),short=v=>dt(v).toLocaleDateString("en-GB",{day:"numeric",month:"short",timeZone:"UTC"}),full=v=>dt(v).toLocaleString("en-GB",{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZone:"UTC",hour12:false})+" UTC";
const cat=v=>v>=1?`C${v}`:({0:"TS","-1":"TD","-2":"SS","-3":"DB","-4":"EX"}[v]||"—");
const assetUrl=path=>new URL(path,assetBaseUrl||document.baseURI).href;

function initMap(){
  map=L.map("map",{zoomControl:true,preferCanvas:true,attributionControl:false});
  map.createPane("geoPane");map.getPane("geoPane").style.zIndex=350;
  map.createPane("sarPane");map.getPane("sarPane").style.zIndex=340;
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{
    maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  map.getContainer().insertAdjacentHTML("beforeend",`<a class="map-credit" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap</a>`);
  const intensityKey=L.control({position:"bottomright"});
  intensityKey.onAdd=()=>{const el=L.DomUtil.create("div","intensity-key");el.innerHTML=`<strong>SAR wind intensity · ${data.sar_color_scale.unit}</strong><i></i><span><b>${data.sar_color_scale.min}</b><b>${data.sar_color_scale.mid}</b><b>${data.sar_color_scale.max}+</b></span>`;return el};
  intensityKey.addTo(map);
  const geostatKey=L.control({position:"bottomright"});
  geostatKey.onAdd=()=>{const el=L.DomUtil.create("div","geostat-key");el.innerHTML=`<strong>Geostationary · ${data.geostat_color_scale.channel}</strong><i></i><span><b>${data.geostat_color_scale.min} K</b><b>${data.geostat_color_scale.mid} K</b><b>${data.geostat_color_scale.max} K</b></span>`;return el};
  geostatKey.addTo(map);
  layers=L.layerGroup().addTo(map);
  geoLayers=L.layerGroup().addTo(map);
  sarLayers=L.layerGroup().addTo(map);
  L.control.layers(null,{"SAR wind fields":sarLayers},{collapsed:false,position:"topright"}).addTo(map);
}
function switcher(){
  $("#stormSwitcher").innerHTML=`<label><span>Storm</span><select id="stormSelect">${data.storms.map(s=>`<option value="${s.id}" ${s.id===storm.id?"selected":""}>${s.id} · ${s.basin}</option>`).join("")}</select></label>`;
  $("#stormSelect").onchange=event=>selectStorm(event.target.value);
}
function renderMap(){
  layers.clearLayers();
  geoLayers.clearLayers();
  sarLayers.clearLayers();
  geoFrames=[];
  sarFrames=[];
  const coords=storm.records.map(r=>[r.lat,r.lon]);
  L.polyline(coords,{color:"#ffffff",weight:8,opacity:.5}).addTo(layers);
  L.polyline(coords,{color:"#e65445",weight:4,opacity:1}).addTo(layers).bindTooltip(`${storm.id} · Storm track`);
  storm.records.filter(r=>r.geo_overlay).forEach(r=>{
    const o=r.geo_overlay;
    const image=L.imageOverlay(assetUrl(o.image),o.bounds,{opacity:1,interactive:true,className:"geo-overlay",pane:"geoPane"})
      .bindTooltip(`<strong>Geostationary · ${o.channel}</strong><br>${full(r.time)}<br>Shared scale: 190–292 K`,{className:"geo-tooltip"});
    const dot=L.circleMarker([r.lat,r.lon],{radius:2.1,color:"#dcebee",weight:1,opacity:.65,fillColor:"#173b48",fillOpacity:.48,riseOnHover:true}).addTo(layers)
      .bindTooltip(`${o.kind} · ${full(r.time)}`);
    const close=L.marker(o.bounds[1],{icon:L.divIcon({className:"geo-close-icon",html:"×",iconSize:[24,24],iconAnchor:[12,12]}),zIndexOffset:2000,title:"Close Geostationary image"});
    const hide=()=>{geoLayers.removeLayer(image);layers.removeLayer(close);dot.setStyle({fillColor:"#173b48",fillOpacity:.48})};
    const show=(showClose=true)=>{image.addTo(geoLayers);if(showClose){close.addTo(layers);close.setZIndexOffset(2000)}else layers.removeLayer(close);dot.setStyle({fillColor:"#f2a654",fillOpacity:1});dot.bringToFront();currentMarker?.bringToFront()};
    close.on("click",hide);
    dot.on("click",()=>geoLayers.hasLayer(image)?hide():show(!$("#animationMode").checked));
    geoFrames.push({record:r,show,hide});
  });
  storm.records.filter(r=>r.sar_overlay).forEach(r=>{
    const o=r.sar_overlay;
    const image=L.imageOverlay(assetUrl(o.image),o.bounds,{opacity:.95,interactive:true,className:"sar-overlay",pane:"sarPane"})
      .bindTooltip(`<strong>SAR-derived WF</strong><br>${full(r.time)}<br>Observed range: ${o.min.toFixed(1)}–${o.max.toFixed(1)} m/s<br>Shared color scale: 0–60+ m/s`,{className:"sar-tooltip"});
    const dot=L.circleMarker([r.lat,r.lon],{radius:5,color:"#263f4a",weight:2,fillColor:"#fff",fillOpacity:1})
      .bindTooltip(`SAR match · ${full(r.time)}`);
    const show=()=>{image.addTo(sarLayers);dot.addTo(sarLayers)};
    const hide=()=>{sarLayers.removeLayer(image);sarLayers.removeLayer(dot)};
    sarFrames.push({record:r,show,hide});
    if(!$("#animationMode").checked)show();
  });
  currentMarker=L.circleMarker(coords[0],{radius:8,color:"#fff",weight:3,fillColor:"#e65445",fillOpacity:1}).addTo(layers);
  map.fitBounds(L.latLngBounds(coords).pad(.3),{animate:false,maxZoom:6});
}
function chartTimeDomain(){
  const stormStart=dt(storm.start).getTime(),stormEnd=dt(storm.end).getTime();
  if(!postProcessing)return[stormStart,stormEnd];
  const sarTimes=storm.records.filter(r=>r.sar).map(r=>dt(r.time).getTime());
  if(!sarTimes.length)return[stormStart,stormEnd];
  const padding=3*3600000;
  return[Math.max(stormStart,sarTimes[0]-padding),Math.min(stormEnd,sarTimes.at(-1)+padding)]
}
function domain(metric,start,end){
  const nwp=metric === "max" ? nwpPoints(start,end).flatMap(series=>series.points.map(point=>point.max)) : [];
  const values=[...storm.records.filter(r=>{const time=dt(r.time).getTime();return time>=start&&time<=end}).flatMap(r=>[predictionValue(metric,r),r.sar?.[metric],metric==="max"?r.ibtracs_msw:null]),...nwp].filter(Number.isFinite);
  if(!values.length)return[0,1];
  const min=Math.min(...values),max=Math.max(...values);
  const padding=Math.max((max-min)*.06,metric==="rmw"?1:.5);
  return[0,Math.max(1,max+padding)]
}
function nwpPoints(start,end){return showNwp?(storm.nwp||[]).map(series=>({...series,points:series.points.filter(point=>{const time=dt(point.time).getTime();return time>=start&&time<=end&&Number.isFinite(point.max)})})).filter(series=>series.points.length):[]}
function median(values){const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!sorted.length)return null;const m=Math.floor(sorted.length/2);return sorted.length%2?sorted[m]:(sorted[m-1]+sorted[m])/2}
function graphPrediction(record){if(graphModel==="model_b")return record.model_b_prediction;if(graphModel==="model_c")return record.model_c_prediction;return record.prediction}
function postprocessExcluded(record){return graphModel==="model_a"&&record.postprocess_excluded}
function smoothedValidValue(metric,record){
  const halfWindow=data.postprocessing.smoothing_hours*3600000/2,target=dt(record.time).getTime();
  return median(storm.records.filter(r=>!postprocessExcluded(r)&&Math.abs(dt(r.time).getTime()-target)<=halfWindow).map(r=>graphPrediction(r)?.[metric]));
}
function predictionValue(metric,record){
  const raw=graphPrediction(record)?.[metric];
  if(!Number.isFinite(raw))return null;
  if(!postProcessing)return raw;
  if(!postprocessExcluded(record))return smoothedValidValue(metric,record);
  const index=storm.records.indexOf(record);let left=index-1,right=index+1;
  while(left>=0&&postprocessExcluded(storm.records[left]))left--;while(right<storm.records.length&&postprocessExcluded(storm.records[right]))right++;
  const a=left>=0?smoothedValidValue(metric,storm.records[left]):null,b=right<storm.records.length?smoothedValidValue(metric,storm.records[right]):null;
  if(!Number.isFinite(a))return b;if(!Number.isFinite(b))return a;
  const ta=dt(storm.records[left].time).getTime(),tb=dt(storm.records[right].time).getTime(),u=(dt(record.time).getTime()-ta)/(tb-ta),blend=u*u*(3-2*u);return a+(b-a)*blend;
}
function chart(metric,def){
  const [start,end]=chartTimeDomain(),[lo,hi]=domain(metric,start,end);
  const x=t=>C.l+(dt(t).getTime()-start)/(end-start)*(C.w-C.l-C.r),y=v=>Math.max(C.t,Math.min(C.h-C.b,C.t+(hi-v)/(hi-lo)*(C.h-C.t-C.b)));
  const inWindow=r=>{const time=dt(r.time).getTime();return time>=start&&time<=end};
  const pred=storm.records.filter(inWindow).map(r=>({...r,plot:predictionValue(metric,r)})).filter(r=>Number.isFinite(r.plot)),sar=storm.records.filter(r=>inWindow(r)&&Number.isFinite(r.sar?.[metric])),ibtracs=metric==="max"?storm.records.filter(r=>inWindow(r)&&Number.isFinite(r.ibtracs_msw)):[],nwp=metric==="max"?nwpPoints(start,end):[];
  const headingKey=metric==="max"?`<span class="category-key">${CATEGORIES.map(c=>`<b style="--cat:${c.color}">${c.label} ${Math.round(c.value)}</b>`).join("")}</span>`:`<span>${def.note||""} · ${def.unit}</span>`;
  const path=(rows,get,breakGaps=false)=>rows.map((r,i)=>`${i&&!(breakGaps&&dt(r.time)-dt(rows[i-1].time)>data.postprocessing.smoothing_hours*3600000)?"L":"M"}${x(r.time).toFixed(1)},${y(get(r)).toFixed(1)}`).join(" ");
  const card=document.createElement("article");card.className="chart-card";card.innerHTML=`<div class="chart-heading"><h3>${def.label}</h3>${headingKey}</div><div class="chart"><svg viewBox="0 0 ${C.w} ${C.h}" preserveAspectRatio="none"></svg></div>`;
  const s=card.querySelector("svg");
  if(metric==="max")CATEGORIES.forEach(c=>{if(c.value>=lo&&c.value<=hi)s.append(svg("line",{class:"category-threshold",x1:C.l,y1:y(c.value),x2:C.w-C.r,y2:y(c.value),stroke:c.color}))});
  [hi,(lo+hi)/2,lo].forEach(v=>{const yp=y(v);s.append(svg("line",{class:"chart-grid",x1:C.l,y1:yp,x2:C.w-C.r,y2:yp}));const t=svg("text",{class:"chart-axis",x:2,y:yp+3});t.textContent=v>=100?Math.round(v):v.toFixed(v<10?1:0);s.append(t)});
  [[short(start),C.l],[short(end),C.w-36]].forEach(([v,xp])=>{const t=svg("text",{class:"chart-axis",x:xp,y:C.h-2});t.textContent=v;s.append(t)});
  nwp.forEach((series,index)=>s.append(svg("path",{class:"nwp-path",d:path(series.points,r=>r.max),"stroke-dasharray":NWP_DASHES[index%NWP_DASHES.length],"aria-label":series.label})));
  s.append(svg("path",{class:"geo-path",d:path(pred,r=>r.plot,postProcessing)}));
  if(pred.length===1)s.append(svg("circle",{class:"geo-dot",cx:x(pred[0].time),cy:y(pred[0].plot),r:3.2}));
  if(sar.length)s.append(svg("path",{class:"sar-path",d:path(sar,r=>r.sar[metric])}));
  sar.forEach(r=>s.append(svg("circle",{class:"sar-dot",cx:x(r.time),cy:y(r.sar[metric]),r:3.5})));
  if(ibtracs.length)s.append(svg("path",{class:"ibtracs-path",d:path(ibtracs,r=>r.ibtracs_msw)}));
  s.append(svg("line",{class:"cursor-line","data-start":start,"data-end":end,x1:C.l,x2:C.l,y1:C.t,y2:C.h-C.b}));
  card.querySelector(".chart").onpointermove=e=>{const rect=e.currentTarget.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((e.clientX-rect.left)/rect.width*C.w-C.l)/(C.w-C.l-C.r))),target=start+fraction*(end-start),i=storm.records.reduce((best,r,index)=>Math.abs(dt(r.time).getTime()-target)<Math.abs(dt(storm.records[best].time).getTime()-target)?index:best,0);$("#timeSlider").value=i;current();const r=storm.records[i],sv=r.sar?.[metric],pv=predictionValue(metric,r),raw=graphPrediction(r)?.[metric],tip=$("#tooltip"),processedLabel=postprocessExcluded(r)?"interpolated":"smoothed",predictionLabel=Number.isFinite(pv)?pv.toFixed(1)+" "+def.unit+(postProcessing?" ("+processedLabel+(Number.isFinite(raw)?"; raw "+raw.toFixed(1):"")+")":""):"Unavailable",ibtracsLabel=metric==="max"&&Number.isFinite(r.ibtracs_msw)?`<br>IBTrACS ${r.ibtracs_msw.toFixed(1)} m/s · ${cat(r.category)}`:"";tip.innerHTML=`<strong>${full(r.time)}</strong><br>${data.models[graphModel].label} ${predictionLabel}<br>SAR-derived WF ${Number.isFinite(sv)?sv.toFixed(1)+" "+def.unit:"No observation"}${ibtracsLabel}`;tip.style.display="block";tip.style.left=Math.min(e.clientX+12,innerWidth-150)+"px";tip.style.top=e.clientY-45+"px"};
  card.querySelector(".chart").onpointerleave=()=>$("#tooltip").style.display="none";return card
}
function charts(){
  $("#charts").replaceChildren(...DISPLAY_METRICS.filter(m=>data.models[graphModel].metrics.includes(m)).map(m=>chart(m,data.metrics[m])));
  if(!$("#tooltip")){const t=document.createElement("div");t.id="tooltip";t.className="tooltip";document.body.append(t)}
}
function current(){
  const i=+$("#timeSlider").value,r=storm.records[i];currentMarker.setLatLng([r.lat,r.lon]);
  $("#currentDate").textContent=full(r.time);$("#observationCount").textContent=`Observation ${i+1} of ${storm.records.length}`;
  $("#categoryValue").textContent=cat(r.category);
  const time=dt(r.time).getTime();$$(".cursor-line").forEach(l=>{const start=+l.dataset.start,end=+l.dataset.end,visible=time>=start&&time<=end,xp=C.l+(time-start)/(end-start)*(C.w-C.l-C.r);l.style.display=visible?"":"none";if(visible){l.setAttribute("x1",xp);l.setAttribute("x2",xp)}})
}
function stop(){clearInterval(timer);timer=null;$("#playButton").textContent="▶"}
function resetAnimation(){geoFrames.forEach(frame=>frame.hide());sarFrames.forEach(frame=>frame.hide())}
function showAnimationFrame(record){geoFrames.find(frame=>frame.record===record)?.show(false);sarFrames.find(frame=>frame.record===record)?.show()}
function restoreManualLayers(){resetAnimation();sarFrames.forEach(frame=>frame.show())}
function setAnimationLayerOrder(enabled){map.getPane("sarPane").style.zIndex=enabled?360:340}
function play(){if(timer)return stop();if($("#animationMode").checked)showAnimationFrame(storm.records[+$("#timeSlider").value]);$("#playButton").textContent="Ⅱ";timer=setInterval(()=>{const s=$("#timeSlider"),next=(+s.value+1)%storm.records.length;if($("#animationMode").checked&&next===0)resetAnimation();s.value=next;current();if($("#animationMode").checked)showAnimationFrame(storm.records[next])},+$("#speedSelector").value)}
function selectStorm(id){
  stop();storm=data.storms.find(s=>s.id===id);switcher();renderMap();
  $("#basinLabel").textContent=`${storm.basin} · 2025`;$("#stormTitle").textContent=storm.id;$("#geoCount").textContent=storm.records.length;$("#sarCount").textContent=storm.sar_matches;
  const sl=$("#timeSlider");sl.max=storm.records.length-1;sl.value=0;$("#startDate").textContent=short(storm.start);$("#midDate").textContent=short(storm.records[Math.floor(storm.records.length/2)].time);$("#endDate").textContent=short(storm.end);charts();current()
}
$("#timeSlider").oninput=()=>{stop();current()};$("#playButton").onclick=play;$("#speedSelector").onchange=()=>{if(timer){stop();play()}};
$("#animationMode").onchange=event=>{setAnimationLayerOrder(event.target.checked);event.target.checked?resetAnimation():restoreManualLayers()};
$("#modelSelector").onchange=event=>{graphModel=event.target.value;$("#predictionLegend").textContent=data.models[graphModel].label;charts();current()};
$("#showNwp").onchange=event=>{showNwp=event.target.checked;$("#nwpLegend").hidden=!showNwp;charts();current()};
$("#postProcessing").onchange=event=>{postProcessing=event.target.checked;charts();current()};
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
