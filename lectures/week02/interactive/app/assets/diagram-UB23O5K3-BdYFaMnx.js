import"./purify.es-Bwm8T4QC.js";import{a as v}from"./chunk-NSK5VX7P-bxC5WUDr.js";import"./src-DRYM6eUR.js";import{n as c}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{t as b}from"./src-Be2gSjc_.js";import{D as S,H as I,K as E,U as F,a as R,b as C,c as z,f as D,v as P,w as B,y as G}from"./chunk-I66GZJ75-CgJN5WrL.js";import{t as V}from"./chunk-3NCLNEKW-D7adDEkj.js";import"./dist-BY5C0xw-.js";import"./chunk-KEIR6QF5-Dy1GNnFZ.js";import"./chunk-MOZMSUNE-C-RMDvp_.js";import"./chunk-OSBZ3O6U-DEYDoDYT.js";import"./chunk-5JV3BV7I-BEi9dboc.js";import"./chunk-CYSBUYHQ-DAWZpY6r.js";import"./chunk-BIQX33UG-B1y9dy0P.js";import"./chunk-EMLP6XTP-Bj2I2iEi.js";import"./chunk-YOTPTUD7-Nm5N45o4.js";import"./chunk-QBLGF6JB-CBncH5PS.js";import"./chunk-5TONJI2A-C5JJwzzL.js";import"./chunk-5HE753X5-DRFzJ7WN.js";import"./chunk-U6XO7XAA-DWWxQ-OA.js";import"./chunk-JG7HCLWE-BEfUAeKb.js";import"./chunk-CQNSW5MT-BOOyLCfW.js";import"./chunk-R7FJI6CG-DWxNgy5y.js";import"./chunk-5FCAYU7R-BJ04i5uO.js";import{t as W}from"./chunk-JWPE2WC7-C-mBVSJi.js";import{n as j}from"./mermaid-parser.core-C1pvq6V4.js";var x={showLegend:!0,ticks:5,max:null,min:0,graticule:"circle"},y=32,M={axes:[],curves:[],options:x},m=structuredClone(M),H=D.radar,U=c(()=>v({...H,...C().radar}),"getConfig"),L=c(()=>m.axes,"getAxes"),_=c(()=>m.curves,"getCurves"),K=c(()=>m.options,"getOptions"),N=c(e=>{m.axes=e.map(t=>({name:t.name,label:t.label??t.name}))},"setAxes"),Z=c(e=>{m.curves=e.map(t=>({name:t.name,label:t.label??t.name,entries:q(t.entries)}))},"setCurves"),q=c(e=>{if(e[0].axis==null)return e.map(a=>a.value);let t=L();if(t.length===0)throw Error("Axes must be populated before curves for reference entries");return t.map(a=>{let r=e.find(i=>{var s;return((s=i.axis)==null?void 0:s.$refText)===a.name});if(r===void 0)throw Error("Missing entry for axis "+a.label);return r.value})},"computeCurveEntries"),h={getAxes:L,getCurves:_,getOptions:K,setAxes:N,setCurves:Z,setOptions:c(e=>{var a,r,i,s,l;let t=e.reduce((o,n)=>(o[n.name]=n,o),{});m.options={showLegend:((a=t.showLegend)==null?void 0:a.value)??x.showLegend,ticks:((r=t.ticks)==null?void 0:r.value)??x.ticks,max:((i=t.max)==null?void 0:i.value)??x.max,min:((s=t.min)==null?void 0:s.value)??x.min,graticule:((l=t.graticule)==null?void 0:l.value)??x.graticule},m.options.ticks>y&&(b.warn(`Radar diagram ticks (${m.options.ticks}) exceeds maximum allowed (${y}). Using ${y} instead.`),m.options.ticks=y)},"setOptions"),getConfig:U,clear:c(()=>{R(),m=structuredClone(M)},"clear"),setAccTitle:F,getAccTitle:G,setDiagramTitle:E,getDiagramTitle:B,getAccDescription:P,setAccDescription:I},J=c(e=>{W(e,h);let{axes:t,curves:a,options:r}=e;h.setAxes(t),h.setCurves(a),h.setOptions(r)},"populate"),Q={parse:c(async e=>{let t=await j("radar",e);b.debug(t),J(t)},"parse")},X=c((e,t,a,r)=>{let i=r.db,s=i.getAxes(),l=i.getCurves(),o=i.getOptions(),n=i.getConfig(),d=i.getDiagramTitle(),p=Y(V(t),n),g=o.max??Math.max(...l.map($=>Math.max(...$.entries))),u=o.min,f=Math.min(n.width,n.height)/2;tt(p,s,f,o.ticks,o.graticule),at(p,s,f,n),k(p,s,l,u,g,o.graticule,n),O(p,l,o.showLegend,n),p.append("text").attr("class","radarTitle").text(d).attr("x",0).attr("y",-n.height/2-n.marginTop)},"draw"),Y=c((e,t)=>{let a=t.width+t.marginLeft+t.marginRight,r=t.height+t.marginTop+t.marginBottom,i={x:t.marginLeft+t.width/2,y:t.marginTop+t.height/2};return z(e,r,a,t.useMaxWidth??!0),e.attr("viewBox",`0 0 ${a} ${r}`).attr("overflow","visible"),e.append("g").attr("transform",`translate(${i.x}, ${i.y})`)},"drawFrame"),tt=c((e,t,a,r,i)=>{if(i==="circle")for(let s=0;s<r;s++){let l=a*(s+1)/r;e.append("circle").attr("r",l).attr("class","radarGraticule")}else if(i==="polygon"){let s=t.length;for(let l=0;l<r;l++){let o=a*(l+1)/r,n=t.map((d,p)=>{let g=2*p*Math.PI/s-Math.PI/2;return`${o*Math.cos(g)},${o*Math.sin(g)}`}).join(" ");e.append("polygon").attr("points",n).attr("class","radarGraticule")}}},"drawGraticule"),at=c((e,t,a,r)=>{let i=t.length;for(let s=0;s<i;s++){let l=t[s].label,o=2*s*Math.PI/i-Math.PI/2,n=Math.cos(o),d=Math.sin(o);e.append("line").attr("x1",0).attr("y1",0).attr("x2",a*r.axisScaleFactor*n).attr("y2",a*r.axisScaleFactor*d).attr("class","radarAxisLine");let p=n>.01?"start":n<-.01?"end":"middle",g=d>.01?"hanging":d<-.01?"auto":"central";e.append("text").text(l).attr("x",a*r.axisLabelFactor*n+4*n).attr("y",a*r.axisLabelFactor*d+4*d).attr("text-anchor",p).attr("dominant-baseline",g).attr("class","radarAxisLabel")}},"drawAxes");function k(e,t,a,r,i,s,l){let o=t.length,n=Math.min(l.width,l.height)/2;a.forEach((d,p)=>{if(d.entries.length!==o)return;let g=d.entries.map((u,f)=>{let $=2*Math.PI*f/o-Math.PI/2,w=T(u,r,i,n);return{x:w*Math.cos($),y:w*Math.sin($)}});s==="circle"?e.append("path").attr("d",A(g,l.curveTension)).attr("class",`radarCurve-${p}`):s==="polygon"&&e.append("polygon").attr("points",g.map(u=>`${u.x},${u.y}`).join(" ")).attr("class",`radarCurve-${p}`)})}c(k,"drawCurves");function T(e,t,a,r){return r*(Math.min(Math.max(e,t),a)-t)/(a-t)}c(T,"relativeRadius");function A(e,t){let a=e.length,r=`M${e[0].x},${e[0].y}`;for(let i=0;i<a;i++){let s=e[(i-1+a)%a],l=e[i],o=e[(i+1)%a],n=e[(i+2)%a],d={x:l.x+(o.x-s.x)*t,y:l.y+(o.y-s.y)*t},p={x:o.x-(n.x-l.x)*t,y:o.y-(n.y-l.y)*t};r+=` C${d.x},${d.y} ${p.x},${p.y} ${o.x},${o.y}`}return`${r} Z`}c(A,"closedRoundCurve");function O(e,t,a,r){if(!a)return;let i=(r.width/2+r.marginRight)*3/4,s=-(r.height/2+r.marginTop)*3/4;t.forEach((l,o)=>{let n=e.append("g").attr("transform",`translate(${i}, ${s+o*20})`);n.append("rect").attr("width",12).attr("height",12).attr("class",`radarLegendBox-${o}`),n.append("text").attr("x",16).attr("y",0).attr("class","radarLegendText").text(l.label)})}c(O,"drawLegend");var et={draw:X},rt=c((e,t)=>{let a="";for(let r=0;r<e.THEME_COLOR_LIMIT;r++){let i=e[`cScale${r}`];a+=`
		.radarCurve-${r} {
			color: ${i};
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
			stroke-width: ${t.curveStrokeWidth};
		}
		.radarLegendBox-${r} {
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
		}
		`}return a},"genIndexStyles"),it=c(e=>{let t=v(S(),C().themeVariables);return{themeVariables:t,radarOptions:v(t.radar,e)}},"buildRadarStyleOptions"),st={parser:Q,db:h,renderer:et,styles:c(({radar:e}={})=>{let{themeVariables:t,radarOptions:a}=it(e);return`
	.radarTitle {
		font-size: ${t.fontSize};
		color: ${t.titleColor};
		dominant-baseline: hanging;
		text-anchor: middle;
	}
	.radarAxisLine {
		stroke: ${a.axisColor};
		stroke-width: ${a.axisStrokeWidth};
	}
	.radarAxisLabel {
		font-size: ${a.axisLabelFontSize}px;
		color: ${a.axisColor};
	}
	.radarGraticule {
		fill: ${a.graticuleColor};
		fill-opacity: ${a.graticuleOpacity};
		stroke: ${a.graticuleColor};
		stroke-width: ${a.graticuleStrokeWidth};
	}
	.radarLegendText {
		text-anchor: start;
		font-size: ${a.legendFontSize}px;
		dominant-baseline: hanging;
	}
	${rt(t,a)}
	`},"styles")};export{st as diagram};
