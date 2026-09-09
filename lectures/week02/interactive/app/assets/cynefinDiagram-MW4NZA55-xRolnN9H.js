import"./purify.es-Bwm8T4QC.js";import{a as G}from"./chunk-NSK5VX7P-bxC5WUDr.js";import"./src-DRYM6eUR.js";import{n as o}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{t as V}from"./src-Be2gSjc_.js";import{D as et,H as yt,K as xt,U as ht,a as gt,b as K,c as $t,f as ut,v as bt,w as wt,y as Ct}from"./chunk-I66GZJ75-CgJN5WrL.js";import{t as kt}from"./chunk-3NCLNEKW-D7adDEkj.js";import"./dist-BY5C0xw-.js";import"./chunk-KEIR6QF5-Dy1GNnFZ.js";import"./chunk-MOZMSUNE-C-RMDvp_.js";import"./chunk-OSBZ3O6U-DEYDoDYT.js";import"./chunk-5JV3BV7I-BEi9dboc.js";import"./chunk-CYSBUYHQ-DAWZpY6r.js";import"./chunk-BIQX33UG-B1y9dy0P.js";import"./chunk-EMLP6XTP-Bj2I2iEi.js";import"./chunk-YOTPTUD7-Nm5N45o4.js";import"./chunk-QBLGF6JB-CBncH5PS.js";import"./chunk-5TONJI2A-C5JJwzzL.js";import"./chunk-5HE753X5-DRFzJ7WN.js";import"./chunk-U6XO7XAA-DWWxQ-OA.js";import"./chunk-JG7HCLWE-BEfUAeKb.js";import"./chunk-CQNSW5MT-BOOyLCfW.js";import"./chunk-R7FJI6CG-DWxNgy5y.js";import"./chunk-5FCAYU7R-BJ04i5uO.js";import{t as Dt}from"./chunk-JWPE2WC7-C-mBVSJi.js";import{n as Bt}from"./mermaid-parser.core-C1pvq6V4.js";var at=o(()=>({domains:new Map,transitions:[]}),"createDefaultData"),W=at(),_={getDomains:o(()=>W.domains,"getDomains"),getTransitions:o(()=>W.transitions,"getTransitions"),setDomains:o(t=>{if(t)for(let e of t){let a=e.domain,r=(e.items??[]).map(l=>({label:l.label}));W.domains.set(a,{name:a,items:r})}},"setDomains"),setTransitions:o(t=>{t&&(W.transitions=t.filter(e=>e.from===e.to?(V.warn(`Cynefin: self-loop transition on domain "${e.from}" is not meaningful and will be skipped.`),!1):!0).map(e=>({from:e.from,to:e.to,label:e.label||void 0})))},"setTransitions"),getConfig:o(()=>G({...ut.cynefin,...K().cynefin}),"getConfig"),clear:o(()=>{gt(),W=at()},"clear"),setAccTitle:ht,getAccTitle:Ct,setDiagramTitle:xt,getDiagramTitle:wt,getAccDescription:bt,setAccDescription:yt},At=o(t=>{Dt(t,_),_.setDomains(t.domains),_.setTransitions(t.transitions)},"populate"),Tt={parse:o(async t=>{let e=await Bt("cynefin",t);V.debug(e),At(e)},"parse")};function E(t){let e=t+1831565813|0;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}o(E,"seededRandom");function rt(t){let e=0;for(let a=0;a<t.length;a++){let r=t.charCodeAt(a);e=(e<<5)-e+r,e|=0}return e}o(rt,"hashString");function nt(t,e){return typeof t=="number"&&Number.isFinite(t)&&t!==0?t:rt(e)}o(nt,"resolveSeed");function it(t,e,a,r){let l=t/2,p=r??t*.015,F=e/7,d=[];for(let n=0;n<=7;n++){let c=E(a+n*17)*p*2-p;d.push({x:l+c,y:n*F})}let D=`M${d[0].x},${d[0].y}`;for(let n=0;n<d.length-1;n++){let c=d[n],m=d[n+1],f=(c.y+m.y)/2,y=n%2==0?1:-1,g=p*1.5*y*E(a+n*31+7),w=c.x+g,I=f,R=m.x-g;D+=` C${w},${I} ${R},${f} ${m.x},${m.y}`}return D}o(it,"generateFoldPath");function ot(t,e,a,r){let l=e/2,p=r??e*.015,F=t/7,d=[];for(let n=0;n<=7;n++){let c=E(a+n*23)*p*2-p;d.push({x:n*F,y:l+c})}let D=`M${d[0].x},${d[0].y}`;for(let n=0;n<d.length-1;n++){let c=d[n],m=d[n+1],f=(c.x+m.x)/2,y=n%2==0?1:-1,g=p*1.5*y*E(a+n*37+11),w=f,I=c.y+g,R=f,N=m.y-g;D+=` C${w},${I} ${R},${N} ${m.x},${m.y}`}return D}o(ot,"generateHorizontalBoundary");function lt(t,e){let a=t/2,r=e*.5,l=e,p=t*.03;return[`M${a},${r}`,`C${a+p},${r+(l-r)*.2}`,`${a-p*1.5},${r+(l-r)*.55}`,`${a+p*.5},${r+(l-r)*.75}`,`C${a-p},${r+(l-r)*.85}`,`${a+p*.3},${r+(l-r)*.95}`,`${a},${l}`].join(" ")}o(lt,"generateCliffPath");function st(t,e,a,r){return[`M${t-a},${e}`,`A${a},${r} 0 1,1 ${t+a},${e}`,`A${a},${r} 0 1,1 ${t-a},${e}`,"Z"].join(" ")}o(st,"generateConfusionPath");var ct={complex:{model:"Probe \u2192 Sense \u2192 Respond",practice:"Emergent Practices"},complicated:{model:"Sense \u2192 Analyse \u2192 Respond",practice:"Good Practices"},clear:{model:"Sense \u2192 Categorise \u2192 Respond",practice:"Best Practices"},chaotic:{model:"Act \u2192 Sense \u2192 Respond",practice:"Novel Practices"},confusion:{model:"",practice:"Disorder"}},St=o((t,e)=>{let a=t/2,r=e/2;return{complex:{cx:a/2,cy:r/2,x:0,y:0,w:a,h:r},complicated:{cx:a+a/2,cy:r/2,x:a,y:0,w:a,h:r},chaotic:{cx:a/2,cy:r+r/2,x:0,y:r,w:a,h:r},clear:{cx:a+a/2,cy:r+r/2,x:a,y:r,w:a,h:r},confusion:{cx:a,cy:r,x:a*.7,y:r*.7,w:a*.6,h:r*.6}}},"getDomainLayouts"),vt=o(()=>G(et(),K().themeVariables).cynefin,"getCynefinDomainColors"),Q=3,zt={draw:o((t,e,a,r)=>{let l=r.db,p=l.getDomains(),F=l.getTransitions(),d=l.getDiagramTitle(),D=l.getAccTitle(),n=l.getAccDescription(),c=l.getConfig(),m=vt();V.debug("Rendering Cynefin diagram");let f=c.width,y=c.height,g=c.padding,w=c.showDomainDescriptions,I=c.boundaryAmplitude,R=f+g*2,N=y+g*2,j={complex:m.complexBg,complicated:m.complicatedBg,clear:m.clearBg,chaotic:m.chaoticBg,confusion:m.confusionBg},B=kt(e);$t(B,N,R,c.useMaxWidth??!0),B.attr("viewBox",`0 0 ${R} ${N}`),D&&B.append("title").text(D),n&&B.append("desc").text(n);let A=B.append("g").attr("transform",`translate(${g}, ${g})`),H=St(f,y),X=nt(c.seed,e),dt=A.append("g").attr("class","cynefin-backgrounds"),U=["complex","complicated","chaotic","clear"];for(let s of U){let i=H[s];dt.append("rect").attr("class","cynefinDomain").attr("x",i.x).attr("y",i.y).attr("width",i.w).attr("height",i.h).attr("fill",j[s]).attr("fill-opacity",.4).attr("stroke","none")}let q=A.append("g").attr("class","cynefin-boundaries");q.append("path").attr("class","cynefinBoundary").attr("d",it(f,y,X,I)).attr("fill","none"),q.append("path").attr("class","cynefinBoundary").attr("d",ot(f,y,X+100,I)).attr("fill","none"),q.append("path").attr("class","cynefinCliff").attr("d",lt(f,y)).attr("fill","none");let ft=f*.15,pt=y*.15;A.append("path").attr("class","cynefinConfusion").attr("d",st(f/2,y/2,ft,pt)).attr("fill",j.confusion).attr("fill-opacity",.5);let Y=A.append("g").attr("class","cynefin-labels");for(let s of U){let i=H[s];Y.append("text").attr("class","cynefinDomainLabel").attr("x",i.cx).attr("y",w?i.cy-30:i.cy).attr("text-anchor","middle").attr("dominant-baseline","middle").text(s.charAt(0).toUpperCase()+s.slice(1))}if(Y.append("text").attr("class","cynefinDomainLabel").attr("x",f/2).attr("y",w?y/2-10:y/2).attr("text-anchor","middle").attr("dominant-baseline","middle").text("Confusion"),w){let s=A.append("g").attr("class","cynefin-subtitles");for(let i of U){let h=H[i],x=ct[i];s.append("text").attr("class","cynefinSubtitle").attr("x",h.cx).attr("y",h.cy-10).attr("text-anchor","middle").attr("dominant-baseline","middle").text(x.model),s.append("text").attr("class","cynefinSubtitle").attr("x",h.cx).attr("y",h.cy+5).attr("text-anchor","middle").attr("dominant-baseline","middle").text(x.practice)}s.append("text").attr("class","cynefinSubtitle").attr("x",f/2).attr("y",y/2+8).attr("text-anchor","middle").attr("dominant-baseline","middle").text(ct.confusion.practice)}let Z=A.append("g").attr("class","cynefin-items");for(let s of["complex","complicated","chaotic","clear","confusion"]){let i=p.get(s);if(!i||i.items.length===0)continue;let h=H[s],x=s==="confusion",z=i.items,M=0;x&&i.items.length>Q&&(M=i.items.length-Q,z=i.items.slice(0,Q));let T;if(x){let u=w?22:14;T=h.cy+u}else T=h.cy+(w?25:15);if([...z].forEach((u,S)=>{let C=T+S*30,v=Z.append("g"),L=v.append("text").attr("class","cynefinItemText").attr("x",0).attr("y",26/2).attr("text-anchor","middle").attr("dominant-baseline","central").text(u.label),b=u.label.length*7,$=L.node();if($&&typeof $.getBBox=="function"){let O=$.getBBox();O.width>0&&(b=O.width)}let k=b+20,P=h.cx-k/2;v.attr("transform",`translate(${P}, ${C})`),v.insert("rect","text").attr("class","cynefinItem").attr("x",0).attr("y",0).attr("width",k).attr("height",26).attr("rx",4).attr("ry",4).attr("fill",j[s]).attr("fill-opacity",.95),L.attr("x",k/2).attr("y",26/2)}),M>0){let u=T+z.length*30,S=`+${M} more`,C=Z.append("g"),v=C.append("text").attr("class","cynefinItemText").attr("x",0).attr("y",26/2).attr("text-anchor","middle").attr("dominant-baseline","central").text(S),L=S.length*7,b=v.node();if(b&&typeof b.getBBox=="function"){let P=b.getBBox();P.width>0&&(L=P.width)}let $=L+20,k=h.cx-$/2;C.attr("transform",`translate(${k}, ${u})`),C.insert("rect","text").attr("class","cynefinItemOverflow").attr("x",0).attr("y",0).attr("width",$).attr("height",26).attr("rx",4).attr("ry",4).attr("fill",j[s]).attr("fill-opacity",.6),v.attr("x",$/2).attr("y",26/2)}}if(F.length>0){let s=B.select("defs").empty()?B.append("defs"):B.select("defs"),i=`cynefin-arrow-${e}`;s.append("marker").attr("id",i).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto-start-reverse").append("path").attr("d","M 0 0 L 10 5 L 0 10 z").attr("class","cynefinArrowHead");let h=A.append("g").attr("class","cynefin-arrows");F.forEach(x=>{let z=H[x.from],M=H[x.to];if(!z||!M)return;if(x.from===x.to){V.warn(`Cynefin renderer: skipping self-loop on domain "${x.from}"`);return}let T=z.cx,u=z.cy,S=M.cx,C=M.cy,v=(T+S)/2,L=(u+C)/2,b=S-T,$=C-u,k=Math.sqrt(b*b+$*$),P=k*.15,O=-$/k,mt=b/k,J=v+O*P,tt=L+mt*P;h.append("path").attr("class","cynefinArrowLine").attr("d",`M${T},${u} Q${J},${tt} ${S},${C}`).attr("fill","none").attr("marker-end",`url(#${i})`),x.label&&h.append("text").attr("class","cynefinArrowLabel").attr("x",J).attr("y",tt-6).attr("text-anchor","middle").attr("dominant-baseline","auto").text(x.label)})}d&&A.append("text").attr("class","cynefinTitle").attr("x",f/2).attr("y",-g/2).attr("text-anchor","middle").attr("dominant-baseline","middle").text(d)},"draw")},Mt=o(()=>G(et(),K().themeVariables).cynefin,"getCynefinTheme"),Lt={parser:Tt,db:_,renderer:zt,styles:o(()=>{let t=Mt();return`
	.cynefinDomain {
		stroke: none;
	}
	.cynefinDomainLabel {
		font-size: ${t.domainFontSize}px;
		font-weight: bold;
		fill: ${t.labelColor};
	}
	.cynefinSubtitle {
		font-size: ${t.itemFontSize-1}px;
		fill: ${t.textColor};
		font-style: italic;
	}
	.cynefinItem {
		fill-opacity: 0.95;
		stroke: ${t.boundaryColor};
		stroke-width: 1;
	}
	.cynefinItemText {
		font-size: ${t.itemFontSize}px;
		fill: ${t.textColor};
	}
	.cynefinItemOverflow {
		fill-opacity: 0.6;
		stroke: ${t.boundaryColor};
		stroke-width: 1;
		stroke-dasharray: 3 2;
	}
	.cynefinBoundary {
		stroke: ${t.boundaryColor};
		stroke-width: ${t.boundaryWidth};
		stroke-dasharray: 6 3;
	}
	.cynefinCliff {
		stroke: ${t.cliffColor};
		stroke-width: ${t.cliffWidth};
	}
	.cynefinConfusion {
		stroke: ${t.boundaryColor};
		stroke-width: 1.5;
		stroke-dasharray: 4 2;
	}
	.cynefinArrowLine {
		stroke: ${t.arrowColor};
		stroke-width: ${t.arrowWidth};
		fill: none;
	}
	.cynefinArrowHead {
		fill: ${t.arrowColor};
		stroke: none;
	}
	.cynefinArrowLabel {
		font-size: ${t.itemFontSize-1}px;
		fill: ${t.textColor};
	}
	.cynefinTitle {
		font-size: ${t.domainFontSize+2}px;
		font-weight: bold;
		fill: ${t.labelColor};
	}
	`},"styles")};export{Lt as diagram};
