var b,R;import{n as h}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{t as w}from"./src-Be2gSjc_.js";import{D as _,a as G,b as O,c as H,x as U,z as J}from"./chunk-I66GZJ75-CgJN5WrL.js";import{t as K}from"./chunk-3NCLNEKW-D7adDEkj.js";var N="",M="",A="",B=[],v=new Map,F=h(e=>J(e,U()),"sanitizeText"),y=h(e=>{switch(e.type){case"terminal":return{...e,value:F(e.value)};case"nonterminal":return{...e,name:F(e.name)};case"sequence":return{...e,elements:e.elements.map(y)};case"choice":return{...e,alternatives:e.alternatives.map(y)};case"optional":return{...e,element:y(e.element)};case"repetition":return{...e,element:y(e.element),separator:e.separator?y(e.separator):void 0};case"special":return{...e,text:F(e.text)}}},"sanitizeAstNode"),Q=h(()=>{N="",M="",A="",B.length=0,v.clear(),G(),w.debug("[Railroad] Database cleared")},"clear"),W=h(e=>{N=F(e),w.debug("[Railroad] Title set:",e)},"setTitle"),q=h(()=>N,"getTitle"),V={clear:Q,setTitle:W,getTitle:q,addRule:h(e=>{let i={...e,name:F(e.name),definition:y(e.definition),comment:e.comment?F(e.comment):void 0};w.debug("[Railroad] Adding rule:",i.name),v.has(i.name)&&w.warn(`[Railroad] Rule '${i.name}' is already defined. Overwriting.`),B.push(i),v.set(i.name,i)},"addRule"),getRules:h(()=>B,"getRules"),getRule:h(e=>v.get(e),"getRule"),setAccTitle:h(e=>{M=F(e).replace(/^\s+/g,""),w.debug("[Railroad] Accessibility title set:",e)},"setAccTitle"),getAccTitle:h(()=>M,"getAccTitle"),setAccDescription:h(e=>{A=F(e).replace(/\n\s+/g,`
`),w.debug("[Railroad] Accessibility description set:",e)},"setAccDescription"),getAccDescription:h(()=>A,"getAccDescription"),setDiagramTitle:W,getDiagramTitle:q},f={compactMode:!1,padding:10,verticalSeparation:8,horizontalSeparation:10,arcRadius:10,fontSize:14,fontFamily:"monospace",terminalFill:"#FFFFC0",terminalStroke:"#000000",terminalTextColor:"#000000",nonTerminalFill:"#FFFFFF",nonTerminalStroke:"#000000",nonTerminalTextColor:"#000000",lineColor:"#000000",strokeWidth:2,markerFill:"#000000",commentFill:"#E8E8E8",commentStroke:"#888888",commentTextColor:"#666666",specialFill:"#F0E0FF",specialStroke:"#8800CC",ruleNameColor:"#000066",showMarkers:!0,markerRadius:5},X=/^#(?:[\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$|^(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\([\d\s%+,./-]+\)$|^[a-z]+$/i,Y=/^[\w "',.-]+$/,Z=new Set(["compactMode","padding","verticalSeparation","horizontalSeparation","arcRadius","fontSize","fontFamily","terminalFill","terminalStroke","terminalTextColor","nonTerminalFill","nonTerminalStroke","nonTerminalTextColor","lineColor","strokeWidth","markerFill","commentFill","commentStroke","commentTextColor","specialFill","specialStroke","ruleNameColor","showMarkers","markerRadius"]),j=h(e=>e?Object.keys(e).every(i=>i==="railroad"||Z.has(i)):!1,"isRailroadStyleOptions"),ee=h(e=>e?"railroad"in e&&e.railroad?e.railroad:j(e)?e:{}:{},"extractRailroadOverrides"),te=h(e=>{if(!e||j(e))return{};let{railroad:i,svgId:t,theme:n,look:a,...r}=e;return r},"extractThemeOverrides"),m=h((e,i)=>{if(typeof e!="string")return i;let t=e.trim();return X.test(t)?t:i},"sanitizeColorValue"),I=h((e,i)=>{if(typeof e!="string")return i;let t=e.trim();return Y.test(t)?t:i},"sanitizeFontFamilyValue"),$=h((e,i)=>{let t=typeof e=="number"?e:typeof e=="string"?Number.parseFloat(e):NaN;return Number.isFinite(t)&&t>=0?t:i},"sanitizeNumberValue"),re=h(e=>{let i=typeof e=="number"?e:typeof e=="string"?Number.parseFloat(e):NaN;return Number.isFinite(i)&&i>0?i:void 0},"parseThemeFontSize"),ie=h(e=>{let i=I(e.fontFamily,f.fontFamily),t=re(e.fontSize)??f.fontSize;return{...f,fontFamily:i,fontSize:t,terminalFill:m(e.secondBkg??e.secondaryColor,f.terminalFill),terminalStroke:m(e.secondaryBorderColor??e.lineColor,f.terminalStroke),terminalTextColor:m(e.secondaryTextColor??e.textColor,f.terminalTextColor),nonTerminalFill:m(e.mainBkg??e.background,f.nonTerminalFill),nonTerminalStroke:m(e.primaryBorderColor??e.lineColor,f.nonTerminalStroke),nonTerminalTextColor:m(e.primaryTextColor??e.textColor,f.nonTerminalTextColor),lineColor:m(e.lineColor,f.lineColor),markerFill:m(e.lineColor,f.markerFill),commentFill:m(e.labelBackground??e.tertiaryColor,f.commentFill),commentStroke:m(e.tertiaryBorderColor??e.lineColor,f.commentStroke),commentTextColor:m(e.tertiaryTextColor??e.textColor,f.commentTextColor),specialFill:m(e.tertiaryColor??e.secondaryColor,f.specialFill),specialStroke:m(e.tertiaryBorderColor??e.secondaryBorderColor,f.specialStroke),ruleNameColor:m(e.titleColor??e.textColor,f.ruleNameColor)}},"buildThemeDefaults"),D=h(e=>{let i=O(),t=ie({..._(),...i.themeVariables??{},...te(e)}),n={...i.railroad??{},...ee(e)};return{compactMode:n.compactMode??t.compactMode,padding:$(n.padding,t.padding),verticalSeparation:$(n.verticalSeparation,t.verticalSeparation),horizontalSeparation:$(n.horizontalSeparation,t.horizontalSeparation),arcRadius:$(n.arcRadius,t.arcRadius),fontSize:$(n.fontSize,t.fontSize),fontFamily:I(n.fontFamily,t.fontFamily),terminalFill:m(n.terminalFill,t.terminalFill),terminalStroke:m(n.terminalStroke,t.terminalStroke),terminalTextColor:m(n.terminalTextColor,t.terminalTextColor),nonTerminalFill:m(n.nonTerminalFill,t.nonTerminalFill),nonTerminalStroke:m(n.nonTerminalStroke,t.nonTerminalStroke),nonTerminalTextColor:m(n.nonTerminalTextColor,t.nonTerminalTextColor),lineColor:m(n.lineColor,t.lineColor),strokeWidth:$(n.strokeWidth,t.strokeWidth),markerFill:m(n.markerFill,t.markerFill),commentFill:m(n.commentFill,t.commentFill),commentStroke:m(n.commentStroke,t.commentStroke),commentTextColor:m(n.commentTextColor,t.commentTextColor),specialFill:m(n.specialFill,t.specialFill),specialStroke:m(n.specialStroke,t.specialStroke),ruleNameColor:m(n.ruleNameColor,t.ruleNameColor),showMarkers:n.showMarkers??t.showMarkers,markerRadius:$(n.markerRadius,t.markerRadius)}},"buildRailroadStyleOptions"),ae=h(e=>{let{fontFamily:i,fontSize:t,terminalFill:n,terminalStroke:a,terminalTextColor:r,nonTerminalFill:o,nonTerminalStroke:g,nonTerminalTextColor:d,lineColor:s,strokeWidth:c,markerFill:l,commentFill:T,commentStroke:p,commentTextColor:u,specialFill:C,specialStroke:S,ruleNameColor:k}=D(e);return`
  .railroad-diagram {
    font-family: ${i};
    font-size: ${t}px;
  }

  .railroad-terminal rect {
    fill: ${n};
    stroke: ${a};
    stroke-width: ${c}px;
  }

  .railroad-terminal text {
    fill: ${r};
    font-family: ${i};
    font-size: ${t}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-nonterminal rect {
    fill: ${o};
    stroke: ${g};
    stroke-width: ${c}px;
  }

  .railroad-nonterminal text {
    fill: ${d};
    font-family: ${i};
    font-size: ${t}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-line {
    stroke: ${s};
    stroke-width: ${c}px;
    fill: none;
  }

  .railroad-start circle,
  .railroad-end circle {
    fill: ${l};
  }

  .railroad-comment ellipse {
    fill: ${T};
    stroke: ${p};
    stroke-width: ${c}px;
  }

  .railroad-comment text {
    fill: ${u};
    font-style: italic;
    font-family: ${i};
    font-size: ${t}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-special rect {
    fill: ${C};
    stroke: ${S};
    stroke-width: ${c}px;
    stroke-dasharray: 5,3;
  }

  .railroad-special text {
    fill: ${d};
    font-family: ${i};
    font-size: ${t}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-rule-name {
    font-weight: bold;
    fill: ${k};
    font-family: ${i};
    font-size: ${t}px;
  }

  .railroad-group {
    /* Grouping container, no specific styles */
  }
`},"getStyles"),x=(b=class{constructor(){this.d=""}moveTo(i,t){return this.d+=`M ${i} ${t} `,this}lineTo(i,t){return this.d+=`L ${i} ${t} `,this}horizontalTo(i){return this.d+=`H ${i} `,this}verticalTo(i){return this.d+=`V ${i} `,this}arcTo(i,t,n,a,r,o,g){return this.d+=`A ${i} ${t} ${n} ${a?1:0} ${r?1:0} ${o} ${g} `,this}build(){return this.d.trim()}},h(b,"PathBuilder"),b),ne=(R=class{constructor(i,t=D()){this.textCache=new Map,this.svg=i,this.config=t}measureText(i){if(this.textCache.has(i))return this.textCache.get(i);let t=this.svg.append("text").attr("font-family",this.config.fontFamily).attr("font-size",this.config.fontSize).text(i),n=t.node().getBBox(),a={width:n.width,height:n.height};return t.remove(),this.textCache.set(i,a),a}renderTerminal(i,t){let n=this.measureText(t),a=n.width+this.config.padding*2,r=n.height+this.config.padding*2,o=i.append("g").attr("class","railroad-terminal");return o.append("rect").attr("x",0).attr("y",0).attr("width",a).attr("height",r).attr("rx",10).attr("ry",10),o.append("text").attr("x",a/2).attr("y",r/2).text(t),{element:o.node(),dimensions:{width:a,height:r,up:r/2,down:r/2}}}renderNonTerminal(i,t){let n=this.measureText(t),a=n.width+this.config.padding*2,r=n.height+this.config.padding*2,o=i.append("g").attr("class","railroad-nonterminal");return o.append("rect").attr("x",0).attr("y",0).attr("width",a).attr("height",r),o.append("text").attr("x",a/2).attr("y",r/2).text(t),{element:o.node(),dimensions:{width:a,height:r,up:r/2,down:r/2}}}renderSequence(i,t){let n=t.map(s=>this.renderExpression(i,s)),a=0,r=0,o=0;for(let s of n)a+=s.dimensions.width,r=Math.max(r,s.dimensions.up),o=Math.max(o,s.dimensions.down);a+=(n.length-1)*this.config.horizontalSeparation;let g=i.append("g").attr("class","railroad-sequence"),d=0;for(let s=0;s<n.length;s++){let c=n[s],l=r-c.dimensions.up;if(g.node().appendChild(c.element).setAttribute("transform",`translate(${d}, ${l})`),s<n.length-1){let T=d+c.dimensions.width,p=T+this.config.horizontalSeparation,u=r;g.append("path").attr("class","railroad-line").attr("d",new x().moveTo(T,u).lineTo(p,u).build())}d+=c.dimensions.width+this.config.horizontalSeparation}return{element:g.node(),dimensions:{width:a,height:r+o,up:r,down:o}}}renderChoice(i,t){let n=t.map(T=>this.renderExpression(i,T)),a=0,r=0;for(let T of n)a=Math.max(a,T.dimensions.width),r+=T.dimensions.height;r+=(n.length-1)*this.config.verticalSeparation;let o=this.config.arcRadius,g=o*4,d=a+g,s=i.append("g").attr("class","railroad-choice"),c=0,l=r/2;for(let T of n){let p=c,u=p+T.dimensions.up,C=o*2+(a-T.dimensions.width)/2;s.node().appendChild(T.element).setAttribute("transform",`translate(${C}, ${p})`);let S=new x,k=u>l;u===l?S.moveTo(0,l).lineTo(C,u):S.moveTo(0,l).arcTo(o,o,0,!1,k,o,l+(k?o:-o)).lineTo(o,u-(k?o:-o)).arcTo(o,o,0,!1,!k,o*2,u).lineTo(C,u),s.append("path").attr("class","railroad-line").attr("d",S.build());let z=new x,E=C+T.dimensions.width,P=d-o*2;u===l?z.moveTo(E,u).lineTo(d,l):z.moveTo(E,u).lineTo(P,u).arcTo(o,o,0,!1,!k,d-o,u+(k?-o:o)).lineTo(d-o,l+(k?o:-o)).arcTo(o,o,0,!1,k,d,l),s.append("path").attr("class","railroad-line").attr("d",z.build()),c+=T.dimensions.height+this.config.verticalSeparation}return{element:s.node(),dimensions:{width:d,height:r,up:l,down:r-l}}}renderOptional(i,t){let n=this.renderExpression(i,t),a=this.config.arcRadius,r=a*2,o=n.dimensions.width+a*4,g=n.dimensions.height+r,d=i.append("g").attr("class","railroad-optional"),s=a*2,c=r;d.node().appendChild(n.element).setAttribute("transform",`translate(${s}, ${c})`);let l=c+n.dimensions.up,T=new x().moveTo(0,l).lineTo(a*2,l);d.append("path").attr("class","railroad-line").attr("d",T.build());let p=new x().moveTo(s+n.dimensions.width,l).lineTo(o,l);d.append("path").attr("class","railroad-line").attr("d",p.build());let u=new x().moveTo(0,l).arcTo(a,a,0,!1,!1,a,l-a).lineTo(a,a).arcTo(a,a,0,!1,!0,a*2,0).lineTo(o-a*2,0).arcTo(a,a,0,!1,!0,o-a,a).lineTo(o-a,l-a).arcTo(a,a,0,!1,!1,o,l);return d.append("path").attr("class","railroad-line").attr("d",u.build()),{element:d.node(),dimensions:{width:o,height:g,up:l,down:g-l}}}renderRepetition(i,t,n){let a=this.renderExpression(i,t),r=this.config.arcRadius,o=r*2,g=a.dimensions.width+r*4,d=n===0,s=a.dimensions.height+o+(d?o:0),c=i.append("g").attr("class","railroad-repetition"),l=r*2,T=d?o:0;c.node().appendChild(a.element).setAttribute("transform",`translate(${l}, ${T})`);let p=T+a.dimensions.up;c.append("path").attr("class","railroad-line").attr("d",new x().moveTo(0,p).lineTo(r*2,p).build()),c.append("path").attr("class","railroad-line").attr("d",new x().moveTo(l+a.dimensions.width,p).lineTo(g,p).build());let u=T+a.dimensions.height+r,C=new x().moveTo(l+a.dimensions.width,p).arcTo(r,r,0,!1,!0,l+a.dimensions.width+r,p+r).lineTo(l+a.dimensions.width+r,u).arcTo(r,r,0,!1,!0,l+a.dimensions.width,u+r).lineTo(r*2,u+r).arcTo(r,r,0,!1,!0,r,u).lineTo(r,p+r).arcTo(r,r,0,!1,!0,r*2,p);if(c.append("path").attr("class","railroad-line").attr("d",C.build()),d){let S=new x().moveTo(0,p).arcTo(r,r,0,!1,!1,r,p-r).lineTo(r,r).arcTo(r,r,0,!1,!0,r*2,0).lineTo(g-r*2,0).arcTo(r,r,0,!1,!0,g-r,r).lineTo(g-r,p-r).arcTo(r,r,0,!1,!1,g,p);c.append("path").attr("class","railroad-line").attr("d",S.build())}return{element:c.node(),dimensions:{width:g,height:s,up:p,down:s-p}}}renderSpecial(i,t){let n=this.measureText("? "+t+" ?"),a=n.width+this.config.padding*2,r=n.height+this.config.padding*2,o=i.append("g").attr("class","railroad-special");return o.append("rect").attr("x",0).attr("y",0).attr("width",a).attr("height",r),o.append("text").attr("x",a/2).attr("y",r/2).text("? "+t+" ?"),{element:o.node(),dimensions:{width:a,height:r,up:r/2,down:r/2}}}renderExpression(i,t){switch(t.type){case"terminal":return this.renderTerminal(i,t.value);case"nonterminal":return this.renderNonTerminal(i,t.name);case"sequence":return this.renderSequence(i,t.elements);case"choice":return this.renderChoice(i,t.alternatives);case"optional":return this.renderOptional(i,t.element);case"repetition":return this.renderRepetition(i,t.element,t.min);case"special":return this.renderSpecial(i,t.text);default:throw Error(`Unknown node type: ${t.type}`)}}renderRule(i,t){let n=this.svg.append("g").attr("class","railroad-rule").attr("transform",`translate(0, ${t})`),a=i.name+" =",r=this.measureText(a).width+20,o=r+20,g=n.append("g"),d=this.renderExpression(g,i.definition),s=Math.max(20,d.dimensions.up),c=s-d.dimensions.up;return g.attr("transform",`translate(${o}, ${c})`),n.append("g").attr("class","railroad-rule-name-group").append("text").attr("class","railroad-rule-name").attr("x",0).attr("y",s).text(a),n.append("g").attr("class","railroad-start").append("circle").attr("cx",r).attr("cy",s).attr("r",this.config.markerRadius),n.append("g").attr("class","railroad-end").append("circle").attr("cx",o+d.dimensions.width+10).attr("cy",s).attr("r",this.config.markerRadius),n.append("path").attr("class","railroad-line").attr("d",new x().moveTo(r+this.config.markerRadius,s).lineTo(o,s).build()),n.append("path").attr("class","railroad-line").attr("d",new x().moveTo(o+d.dimensions.width,s).lineTo(o+d.dimensions.width+10-this.config.markerRadius,s).build()),{height:Math.max(40,c+d.dimensions.height+this.config.padding*2),width:o+d.dimensions.width+10+this.config.markerRadius}}renderDiagram(i){let t=this.config.padding,n=0;for(let a of i){let r=this.renderRule(a,t);t+=r.height+this.config.verticalSeparation,n=Math.max(n,r.width)}return{width:n+this.config.padding*2,height:t+this.config.padding}}},h(R,"RailroadRenderer"),R),L=h((e,i,t)=>{H(e,i.height,i.width,t),e.attr("viewBox",`0 0 ${i.width} ${i.height}`)},"configureRailroadSvgSize"),oe={draw:h((e,i,t)=>{var n;w.debug(`[Railroad] Rendering diagram
`+e);try{let a=K(i);a.attr("class","railroad-diagram");let r=((n=O().railroad)==null?void 0:n.useMaxWidth)??!0,o=V.getRules();if(w.debug(`[Railroad] Rendering ${o.length} rules`),o.length===0){w.warn("[Railroad] No rules to render"),L(a,{height:100,width:200},r);return}L(a,new ne(a,D()).renderDiagram(o),r),w.debug("[Railroad] Render complete")}catch(a){throw w.error("[Railroad] Render error:",a),a}},"draw")};export{ae as n,oe as r,V as t};
