import"./purify.es-Bwm8T4QC.js";import{n as X}from"./ordinal-CSom89lg.js";import{r as w}from"./path-pKytGTsv.js";import{p as B}from"./math-C0ZUp0SY.js";import{t as _}from"./arc-BgDJXxuB.js";import{t as Y}from"./array-C8hY-75m.js";import{a as Z,m as tt}from"./chunk-NSK5VX7P-bxC5WUDr.js";import"./src-DRYM6eUR.js";import{n as c}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{t as E}from"./src-Be2gSjc_.js";import{H as et,K as at,U as rt,a as it,c as nt,f as lt,v as ot,w as st,x as pt,y as ct}from"./chunk-I66GZJ75-CgJN5WrL.js";import{t as ut}from"./chunk-3NCLNEKW-D7adDEkj.js";import"./dist-BY5C0xw-.js";import"./chunk-KEIR6QF5-Dy1GNnFZ.js";import"./chunk-MOZMSUNE-C-RMDvp_.js";import"./chunk-OSBZ3O6U-DEYDoDYT.js";import"./chunk-5JV3BV7I-BEi9dboc.js";import"./chunk-CYSBUYHQ-DAWZpY6r.js";import"./chunk-BIQX33UG-B1y9dy0P.js";import"./chunk-EMLP6XTP-Bj2I2iEi.js";import"./chunk-YOTPTUD7-Nm5N45o4.js";import"./chunk-QBLGF6JB-CBncH5PS.js";import"./chunk-5TONJI2A-C5JJwzzL.js";import"./chunk-5HE753X5-DRFzJ7WN.js";import"./chunk-U6XO7XAA-DWWxQ-OA.js";import"./chunk-JG7HCLWE-BEfUAeKb.js";import"./chunk-CQNSW5MT-BOOyLCfW.js";import"./chunk-R7FJI6CG-DWxNgy5y.js";import"./chunk-5FCAYU7R-BJ04i5uO.js";import{t as mt}from"./chunk-JWPE2WC7-C-mBVSJi.js";import{n as dt}from"./mermaid-parser.core-C1pvq6V4.js";function gt(t,r){return r<t?-1:r>t?1:r>=t?0:NaN}function ht(t){return t}function ft(){var t=ht,r=gt,d=null,s=w(0),p=w(B),S=w(0);function i(e){var l,n=(e=Y(e)).length,x,D,b=0,$=Array(n),g=Array(n),A=+s.apply(this,arguments),m=Math.min(B,Math.max(-B,p.apply(this,arguments)-A)),h,M=Math.min(Math.abs(m)/n,S.apply(this,arguments)),O=M*(m<0?-1:1),u;for(l=0;l<n;++l)(u=g[$[l]=l]=+t(e[l],l,e))>0&&(b+=u);for(r==null?d!=null&&$.sort(function(C,y){return d(e[C],e[y])}):$.sort(function(C,y){return r(g[C],g[y])}),l=0,D=b?(m-n*O)/b:0;l<n;++l,A=h)x=$[l],u=g[x],h=A+(u>0?u*D:0)+O,g[x]={data:e[x],index:l,value:u,startAngle:A,endAngle:h,padAngle:M};return g}return i.value=function(e){return arguments.length?(t=typeof e=="function"?e:w(+e),i):t},i.sortValues=function(e){return arguments.length?(r=e,d=null,i):r},i.sort=function(e){return arguments.length?(d=e,r=null,i):d},i.startAngle=function(e){return arguments.length?(s=typeof e=="function"?e:w(+e),i):s},i.endAngle=function(e){return arguments.length?(p=typeof e=="function"?e:w(+e),i):p},i.padAngle=function(e){return arguments.length?(S=typeof e=="function"?e:w(+e),i):S},i}var I=lt.pie,P={sections:new Map,showData:!1,config:I},F=P.sections,W=P.showData,xt=structuredClone(I),K={getConfig:c(()=>structuredClone(xt),"getConfig"),clear:c(()=>{F=new Map,W=P.showData,it()},"clear"),setDiagramTitle:at,getDiagramTitle:st,setAccTitle:rt,getAccTitle:ct,setAccDescription:et,getAccDescription:ot,addSection:c(({label:t,value:r})=>{if(r<0)throw Error(`"${t}" has invalid value: ${r}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);F.has(t)||(F.set(t,r),E.debug(`added new section: ${t}, with value: ${r}`))},"addSection"),getSections:c(()=>F,"getSections"),setShowData:c(t=>{W=t},"setShowData"),getShowData:c(()=>W,"getShowData")},yt=c((t,r)=>{mt(t,r),r.setShowData(t.showData),t.sections.map(r.addSection)},"populateDb"),vt={parse:c(async t=>{let r=await dt("pie",t);E.debug(r),yt(r,K)},"parse")},wt=c(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieCircle.highlighted{
    scale: 1.05;
    opacity: 1;
  }
  .pieCircle.highlightedOnHover:hover{
    transition-duration: 250ms;
    scale: 1.05;
    opacity: 1;
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),St=c(t=>{let r=[...t.values()].reduce((s,p)=>s+p,0),d=[...t.entries()].map(([s,p])=>({label:s,value:p})).filter(s=>s.value/r*100>=1);return ft().value(s=>s.value).sort(null)(d)},"createPieArcs"),bt={parser:vt,db:K,renderer:{draw:c((t,r,d,s)=>{var U;E.debug(`rendering pie chart
`+t);let p=s.db,S=pt(),i=Z(p.getConfig(),S.pie),e=ut(r),l=e.append("g");l.attr("transform","translate(225,225)");let{themeVariables:n}=S,[x]=tt(n.pieOuterStrokeWidth);x??(x=2);let D=i.legendPosition,b=i.textPosition,$=i.donutHole>0&&i.donutHole<=.9?i.donutHole:0,g=_().innerRadius($*185).outerRadius(185),A=_().innerRadius(185*b).outerRadius(185*b),m=l.append("g");m.append("circle").attr("cx",0).attr("cy",0).attr("r",185+x/2).attr("class","pieOuterCircle");let h=p.getSections(),M=St(h),O=[n.pie1,n.pie2,n.pie3,n.pie4,n.pie5,n.pie6,n.pie7,n.pie8,n.pie9,n.pie10,n.pie11,n.pie12],u=0;h.forEach(a=>{u+=a});let C=M.filter(a=>(a.data.value/u*100).toFixed(0)!=="0"),y=X(O).domain([...h.keys()]);m.selectAll("mySlices").data(C).enter().append("path").attr("d",g).attr("fill",a=>y(a.data.label)).attr("class",a=>{let o="pieCircle";return i.highlightSlice==="hover"?o+=" highlightedOnHover":i.highlightSlice===a.data.label&&(o+=" highlighted"),o}),m.selectAll("mySlices").data(C).enter().append("text").text(a=>(a.data.value/u*100).toFixed(0)+"%").attr("transform",a=>"translate("+A.centroid(a)+")").style("text-anchor","middle").attr("class","slice");let q=l.append("text").text(p.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText"),k=[...h.entries()].map(([a,o])=>({label:a,value:o})),f=l.selectAll(".legend").data(k).enter().append("g").attr("class","legend");f.append("rect").attr("width",18).attr("height",18).style("fill",a=>y(a.label)).style("stroke",a=>y(a.label)),f.append("text").attr("x",22).attr("y",14).text(a=>p.getShowData()?`${a.label} [${a.value}]`:a.label);let T=Math.max(...f.selectAll("text").nodes().map(a=>(a==null?void 0:a.getBoundingClientRect().width)??0)),z=450,H=490,R=k.length*22;switch(D){case"center":f.attr("transform",(a,o)=>{let v=22*k.length/2,N=-T/2-22,Q=o*22-v;return"translate("+N+","+Q+")"});break;case"top":z+=R,f.attr("transform",(a,o)=>`translate(${-T/2-22}, ${o*22-185})`),m.attr("transform",()=>`translate(0, ${R+22})`);break;case"bottom":z+=R,f.attr("transform",(a,o)=>{let v=-T/2-22,N=o*22- -207;return"translate("+v+","+N+")"});break;case"left":H+=22+T,f.attr("transform",(a,o)=>{let v=22*k.length/2;return"translate(-207,"+(o*22-v)+")"}),m.attr("transform",()=>`translate(${T+18+4}, 0)`);break;default:H+=22+T,f.attr("transform",(a,o)=>{let v=22*k.length/2;return"translate(216,"+(o*22-v)+")"});break}let L=((U=q.node())==null?void 0:U.getBoundingClientRect().width)??0,G=450/2-L/2,J=450/2+L/2,V=Math.min(0,G),j=Math.max(H,J)-V;e.attr("viewBox",`${V} 0 ${j} ${z}`),nt(e,z,j,i.useMaxWidth)},"draw")},styles:wt};export{bt as diagram};
