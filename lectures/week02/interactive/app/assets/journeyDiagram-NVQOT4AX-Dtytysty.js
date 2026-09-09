import"./purify.es-Bwm8T4QC.js";import{t as et}from"./arc-BgDJXxuB.js";import{u as it}from"./src-DRYM6eUR.js";import{n as r}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import"./src-Be2gSjc_.js";import{H as gt,K as mt,U as xt,a as kt,c as _t,v as bt,w as vt,x as j,y as $t}from"./chunk-I66GZJ75-CgJN5WrL.js";import"./dist-BY5C0xw-.js";import{t as wt}from"./chunk-5VM5RSS4-CQG8Gkf2.js";import{a as Mt,n as Tt,o as Ct,s as nt}from"./chunk-2GRJ4B5K-B92Uwj7z.js";var X=(function(){var t=r(function(i,a,s,c){for(s||(s={}),c=i.length;c--;s[i[c]]=a);return s},"o"),e=[6,8,10,11,12,14,16,17,18],l=[1,9],y=[1,10],n=[1,11],h=[1,12],u=[1,13],p=[1,14],f={trace:r(function(){},"trace"),yy:{},symbols_:{error:2,start:3,journey:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,title:11,acc_title:12,acc_title_value:13,acc_descr:14,acc_descr_value:15,acc_descr_multiline_value:16,section:17,taskName:18,taskData:19,$accept:0,$end:1},terminals_:{2:"error",4:"journey",6:"EOF",8:"SPACE",10:"NEWLINE",11:"title",12:"acc_title",13:"acc_title_value",14:"acc_descr",15:"acc_descr_value",16:"acc_descr_multiline_value",17:"section",18:"taskName",19:"taskData"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,2]],performAction:r(function(i,a,s,c,d,o,k){var m=o.length-1;switch(d){case 1:return o[m-1];case 2:this.$=[];break;case 3:o[m-1].push(o[m]),this.$=o[m-1];break;case 4:case 5:this.$=o[m];break;case 6:case 7:this.$=[];break;case 8:c.setDiagramTitle(o[m].substr(6)),this.$=o[m].substr(6);break;case 9:this.$=o[m].trim(),c.setAccTitle(this.$);break;case 10:case 11:this.$=o[m].trim(),c.setAccDescription(this.$);break;case 12:c.addSection(o[m].substr(8)),this.$=o[m].substr(8);break;case 13:c.addTask(o[m-1],o[m]),this.$="task";break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:l,12:y,14:n,16:h,17:u,18:p},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:15,11:l,12:y,14:n,16:h,17:u,18:p},t(e,[2,5]),t(e,[2,6]),t(e,[2,8]),{13:[1,16]},{15:[1,17]},t(e,[2,11]),t(e,[2,12]),{19:[1,18]},t(e,[2,4]),t(e,[2,9]),t(e,[2,10]),t(e,[2,13])],defaultActions:{},parseError:r(function(i,a){if(a.recoverable)this.trace(i);else{var s=Error(i);throw s.hash=a,s}},"parseError"),parse:r(function(i){var a=this,s=[0],c=[],d=[null],o=[],k=this.table,m="",b=0,V=0,A=0,pt=2,Z=1,yt=o.slice.call(arguments,1),x=Object.create(this.lexer),I={yy:{}};for(var D in this.yy)Object.prototype.hasOwnProperty.call(this.yy,D)&&(I.yy[D]=this.yy[D]);x.setInput(i,I.yy),I.yy.lexer=x,I.yy.parser=this,x.yylloc===void 0&&(x.yylloc={});var Y=x.yylloc;o.push(Y);var dt=x.options&&x.options.ranges;typeof I.yy.parseError=="function"?this.parseError=I.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ft($){s.length-=2*$,d.length-=$,o.length-=$}r(ft,"popStack");function J(){var $=c.pop()||x.lex()||Z;return typeof $!="number"&&($ instanceof Array&&(c=$,$=c.pop()),$=a.symbols_[$]||$),$}r(J,"lex");for(var _,W,S,v,q,P={},N,T,Q,O;;){if(S=s[s.length-1],this.defaultActions[S]?v=this.defaultActions[S]:(_??(_=J()),v=k[S]&&k[S][_]),v===void 0||!v.length||!v[0]){var tt="";for(N in O=[],k[S])this.terminals_[N]&&N>pt&&O.push("'"+this.terminals_[N]+"'");tt=x.showPosition?"Parse error on line "+(b+1)+`:
`+x.showPosition()+`
Expecting `+O.join(", ")+", got '"+(this.terminals_[_]||_)+"'":"Parse error on line "+(b+1)+": Unexpected "+(_==Z?"end of input":"'"+(this.terminals_[_]||_)+"'"),this.parseError(tt,{text:x.match,token:this.terminals_[_]||_,line:x.yylineno,loc:Y,expected:O})}if(v[0]instanceof Array&&v.length>1)throw Error("Parse Error: multiple actions possible at state: "+S+", token: "+_);switch(v[0]){case 1:s.push(_),d.push(x.yytext),o.push(x.yylloc),s.push(v[1]),_=null,W?(_=W,W=null):(V=x.yyleng,m=x.yytext,b=x.yylineno,Y=x.yylloc,A>0&&A--);break;case 2:if(T=this.productions_[v[1]][1],P.$=d[d.length-T],P._$={first_line:o[o.length-(T||1)].first_line,last_line:o[o.length-1].last_line,first_column:o[o.length-(T||1)].first_column,last_column:o[o.length-1].last_column},dt&&(P._$.range=[o[o.length-(T||1)].range[0],o[o.length-1].range[1]]),q=this.performAction.apply(P,[m,V,b,I.yy,v[1],d,o].concat(yt)),q!==void 0)return q;T&&(s=s.slice(0,-1*T*2),d=d.slice(0,-1*T),o=o.slice(0,-1*T)),s.push(this.productions_[v[1]][0]),d.push(P.$),o.push(P._$),Q=k[s[s.length-2]][s[s.length-1]],s.push(Q);break;case 3:return!0}}return!0},"parse")};f.lexer=(function(){return{EOF:1,parseError:r(function(i,a){if(this.yy.parser)this.yy.parser.parseError(i,a);else throw Error(i)},"parseError"),setInput:r(function(i,a){return this.yy=a||this.yy||{},this._input=i,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:r(function(){var i=this._input[0];return this.yytext+=i,this.yyleng++,this.offset++,this.match+=i,this.matched+=i,i.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),i},"input"),unput:r(function(i){var a=i.length,s=i.split(/(?:\r\n?|\n)/g);this._input=i+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-a),this.offset-=a;var c=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),s.length-1&&(this.yylineno-=s.length-1);var d=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:s?(s.length===c.length?this.yylloc.first_column:0)+c[c.length-s.length].length-s[0].length:this.yylloc.first_column-a},this.options.ranges&&(this.yylloc.range=[d[0],d[0]+this.yyleng-a]),this.yyleng=this.yytext.length,this},"unput"),more:r(function(){return this._more=!0,this},"more"),reject:r(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:r(function(i){this.unput(this.match.slice(i))},"less"),pastInput:r(function(){var i=this.matched.substr(0,this.matched.length-this.match.length);return(i.length>20?"...":"")+i.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:r(function(){var i=this.match;return i.length<20&&(i+=this._input.substr(0,20-i.length)),(i.substr(0,20)+(i.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:r(function(){var i=this.pastInput(),a=Array(i.length+1).join("-");return i+this.upcomingInput()+`
`+a+"^"},"showPosition"),test_match:r(function(i,a){var s,c,d;if(this.options.backtrack_lexer&&(d={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(d.yylloc.range=this.yylloc.range.slice(0))),c=i[0].match(/(?:\r\n?|\n).*/g),c&&(this.yylineno+=c.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:c?c[c.length-1].length-c[c.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+i[0].length},this.yytext+=i[0],this.match+=i[0],this.matches=i,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(i[0].length),this.matched+=i[0],s=this.performAction.call(this,this.yy,this,a,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),s)return s;if(this._backtrack){for(var o in d)this[o]=d[o];return!1}return!1},"test_match"),next:r(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var i,a,s,c;this._more||(this.yytext="",this.match="");for(var d=this._currentRules(),o=0;o<d.length;o++)if(s=this._input.match(this.rules[d[o]]),s&&(!a||s[0].length>a[0].length)){if(a=s,c=o,this.options.backtrack_lexer){if(i=this.test_match(s,d[o]),i!==!1)return i;if(this._backtrack){a=!1;continue}else return!1}else if(!this.options.flex)break}return a?(i=this.test_match(a,d[c]),i===!1?!1:i):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:r(function(){return this.next()||this.lex()},"lex"),begin:r(function(i){this.conditionStack.push(i)},"begin"),popState:r(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:r(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:r(function(i){return i=this.conditionStack.length-1-Math.abs(i||0),i>=0?this.conditionStack[i]:"INITIAL"},"topState"),pushState:r(function(i){this.begin(i)},"pushState"),stateStackSize:r(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:r(function(i,a,s,c){switch(s){case 0:break;case 1:break;case 2:return 10;case 3:break;case 4:break;case 5:return 4;case 6:return 11;case 7:return this.begin("acc_title"),12;case 8:return this.popState(),"acc_title_value";case 9:return this.begin("acc_descr"),14;case 10:return this.popState(),"acc_descr_value";case 11:this.begin("acc_descr_multiline");break;case 12:this.popState();break;case 13:return"acc_descr_multiline_value";case 14:return 17;case 15:return 18;case 16:return 19;case 17:return":";case 18:return 6;case 19:return"INVALID"}},"anonymous"),rules:[/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:journey\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[12,13],inclusive:!1},acc_descr:{rules:[10],inclusive:!1},acc_title:{rules:[8],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,9,11,14,15,16,17,18,19],inclusive:!0}}}})();function g(){this.yy={}}return r(g,"Parser"),g.prototype=f,f.Parser=g,new g})();X.parser=X;var Et=X,F="",G=[],B=[],L=[],It=r(function(){G.length=0,B.length=0,F="",L.length=0,kt()},"clear"),St=r(function(t){F=t,G.push(t)},"addSection"),At=r(function(){return G},"getSections"),Pt=r(function(){let t=st(),e=0;for(;!t&&e<100;)t=st(),e++;return B.push(...L),B},"getTasks"),jt=r(function(){let t=[];return B.forEach(e=>{e.people&&t.push(...e.people)}),[...new Set(t)].sort()},"updateActors"),Ft=r(function(t,e){let l=e.substr(1).split(":"),y=0,n=[];l.length===1?(y=Number(l[0]),n=[]):(y=Number(l[0]),n=l[1].split(","));let h=n.map(p=>p.trim()),u={section:F,type:F,people:h,task:t,score:y};L.push(u)},"addTask"),Vt=r(function(t){let e={section:F,type:F,description:t,task:t,classes:[]};B.push(e)},"addTaskOrg"),st=r(function(){let t=r(function(l){return L[l].processed},"compileTask"),e=!0;for(let[l,y]of L.entries())t(l),e&&(e=y.processed);return e},"compileTasks"),rt={getConfig:r(()=>j().journey,"getConfig"),clear:It,setDiagramTitle:mt,getDiagramTitle:vt,setAccTitle:xt,getAccTitle:$t,setAccDescription:gt,getAccDescription:bt,addSection:St,getSections:At,getTasks:Pt,addTask:Ft,addTaskOrg:Vt,getActors:r(function(){return jt()},"getActors")},Bt=r(t=>`.label {
    font-family: ${t.fontFamily};
    color: ${t.textColor};
  }
  .mouth {
    stroke: #666;
  }

  line {
    stroke: ${t.textColor}
  }

  .legend {
    fill: ${t.textColor};
    font-family: ${t.fontFamily};
  }

  .label text {
    fill: #333;
  }
  .label {
    color: ${t.textColor}
  }

  .face {
    ${t.faceColor?`fill: ${t.faceColor}`:"fill: #FFF8DC"};
    stroke: #999;
  }

  .node rect,
  .node circle,
  .node ellipse,
  .node polygon,
  .node path {
    fill: ${t.mainBkg};
    stroke: ${t.nodeBorder};
    stroke-width: 1px;
  }

  .node .label {
    text-align: center;
  }
  .node.clickable {
    cursor: pointer;
  }

  .arrowheadPath {
    fill: ${t.arrowheadColor};
  }

  .edgePath .path {
    stroke: ${t.lineColor};
    stroke-width: 1.5px;
  }

  .flowchart-link {
    stroke: ${t.lineColor};
    fill: none;
  }

  .edgeLabel {
    background-color: ${t.edgeLabelBackground};
    rect {
      opacity: 0.5;
    }
    text-align: center;
  }

  .cluster rect {
  }

  .cluster text {
    fill: ${t.titleColor};
  }

  div.mermaidTooltip {
    position: absolute;
    text-align: center;
    max-width: 200px;
    padding: 2px;
    font-family: ${t.fontFamily};
    font-size: 12px;
    background: ${t.tertiaryColor};
    border: 1px solid ${t.border2};
    border-radius: 2px;
    pointer-events: none;
    z-index: 100;
  }

  .task-type-0, .section-type-0  {
    ${t.fillType0?`fill: ${t.fillType0}`:""};
  }
  .task-type-1, .section-type-1  {
    ${t.fillType0?`fill: ${t.fillType1}`:""};
  }
  .task-type-2, .section-type-2  {
    ${t.fillType0?`fill: ${t.fillType2}`:""};
  }
  .task-type-3, .section-type-3  {
    ${t.fillType0?`fill: ${t.fillType3}`:""};
  }
  .task-type-4, .section-type-4  {
    ${t.fillType0?`fill: ${t.fillType4}`:""};
  }
  .task-type-5, .section-type-5  {
    ${t.fillType0?`fill: ${t.fillType5}`:""};
  }
  .task-type-6, .section-type-6  {
    ${t.fillType0?`fill: ${t.fillType6}`:""};
  }
  .task-type-7, .section-type-7  {
    ${t.fillType0?`fill: ${t.fillType7}`:""};
  }

  .actor-0 {
    ${t.actor0?`fill: ${t.actor0}`:""};
  }
  .actor-1 {
    ${t.actor1?`fill: ${t.actor1}`:""};
  }
  .actor-2 {
    ${t.actor2?`fill: ${t.actor2}`:""};
  }
  .actor-3 {
    ${t.actor3?`fill: ${t.actor3}`:""};
  }
  .actor-4 {
    ${t.actor4?`fill: ${t.actor4}`:""};
  }
  .actor-5 {
    ${t.actor5?`fill: ${t.actor5}`:""};
  }
  ${wt()}
`,"getStyles"),U=r(function(t,e){return Mt(t,e)},"drawRect"),Lt=r(function(t,e){let l=t.append("circle").attr("cx",e.cx).attr("cy",e.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),y=t.append("g");y.append("circle").attr("cx",e.cx-15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),y.append("circle").attr("cx",e.cx+15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666");function n(p){let f=et().startAngle(Math.PI/2).endAngle(3*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",f).attr("transform","translate("+e.cx+","+(e.cy+2)+")")}r(n,"smile");function h(p){let f=et().startAngle(3*Math.PI/2).endAngle(5*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",f).attr("transform","translate("+e.cx+","+(e.cy+7)+")")}r(h,"sad");function u(p){p.append("line").attr("class","mouth").attr("stroke",2).attr("x1",e.cx-5).attr("y1",e.cy+7).attr("x2",e.cx+5).attr("y2",e.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return r(u,"ambivalent"),e.score>3?n(y):e.score<3?h(y):u(y),l},"drawFace"),at=r(function(t,e){let l=t.append("circle");return l.attr("cx",e.cx),l.attr("cy",e.cy),l.attr("class","actor-"+e.pos),l.attr("fill",e.fill),l.attr("stroke",e.stroke),l.attr("r",e.r),l.class!==void 0&&l.attr("class",l.class),e.title!==void 0&&l.append("title").text(e.title),l},"drawCircle"),ot=r(function(t,e){return Ct(t,e)},"drawText"),Rt=r(function(t,e){function l(n,h,u,p,f){return n+","+h+" "+(n+u)+","+h+" "+(n+u)+","+(h+p-f)+" "+(n+u-f*1.2)+","+(h+p)+" "+n+","+(h+p)}r(l,"genPoints");let y=t.append("polygon");y.attr("points",l(e.x,e.y,50,20,7)),y.attr("class","labelBox"),e.y+=e.labelMargin,e.x+=.5*e.labelMargin,ot(t,e)},"drawLabel"),Nt=r(function(t,e,l){let y=t.append("g"),n=nt();n.x=e.x,n.y=e.y,n.fill=e.fill,n.width=l.width*e.taskCount+l.diagramMarginX*(e.taskCount-1),n.height=l.height,n.class="journey-section section-type-"+e.num,n.rx=3,n.ry=3,U(y,n),lt(l)(e.text,y,n.x,n.y,n.width,n.height,{class:"journey-section section-type-"+e.num},l,e.colour)},"drawSection"),H=-1,Ot=r(function(t,e,l,y){let n=e.x+l.width/2,h=t.append("g");H++,h.append("line").attr("id",y+"-task"+H).attr("x1",n).attr("y1",e.y).attr("x2",n).attr("y2",450).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Lt(h,{cx:n,cy:300+(5-e.score)*30,score:e.score});let u=nt();u.x=e.x,u.y=e.y,u.fill=e.fill,u.width=l.width,u.height=l.height,u.class="task task-type-"+e.num,u.rx=3,u.ry=3,U(h,u);let p=e.x+14;e.people.forEach(f=>{let g=e.actors[f].color;at(h,{cx:p,cy:e.y,r:7,fill:g,stroke:"#000",title:f,pos:e.actors[f].position}),p+=10}),lt(l)(e.task,h,u.x,u.y,u.width,u.height,{class:"task"},l,e.colour)},"drawTask"),zt=r(function(t,e){Tt(t,e)},"drawBackgroundRect"),lt=(function(){function t(n,h,u,p,f,g,i,a){y(h.append("text").attr("x",u+f/2).attr("y",p+g/2+5).style("font-color",a).style("text-anchor","middle").text(n),i)}r(t,"byText");function e(n,h,u,p,f,g,i,a,s){let{taskFontSize:c,taskFontFamily:d}=a,o=n.split(/<br\s*\/?>/gi);for(let k=0;k<o.length;k++){let m=k*c-c*(o.length-1)/2,b=h.append("text").attr("x",u+f/2).attr("y",p).attr("fill",s).style("text-anchor","middle").style("font-size",c).style("font-family",d);b.append("tspan").attr("x",u+f/2).attr("dy",m).text(o[k]),b.attr("y",p+g/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),y(b,i)}}r(e,"byTspan");function l(n,h,u,p,f,g,i,a){let s=h.append("switch"),c=s.append("foreignObject").attr("x",u).attr("y",p).attr("width",f).attr("height",g).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");c.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(n),e(n,s,u,p,f,g,i,a),y(c,i)}r(l,"byFo");function y(n,h){for(let u in h)u in h&&n.attr(u,h[u])}return r(y,"_setTextAttrs"),function(n){return n.textPlacement==="fo"?l:n.textPlacement==="old"?t:e}})(),R={drawRect:U,drawCircle:at,drawSection:Nt,drawText:ot,drawLabel:Rt,drawTask:Ot,drawBackgroundRect:zt,initGraphics:r(function(t,e){H=-1,t.append("defs").append("marker").attr("id",e+"-arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")},"initGraphics")},Dt=r(function(t){Object.keys(t).forEach(function(e){M[e]=t[e]})},"setConf"),C={},z=0;function ct(t){let e=j().journey,l=e.maxLabelWidth;z=0;let y=60;Object.keys(C).forEach(n=>{let h=C[n].color,u={cx:20,cy:y,r:7,fill:h,stroke:"#000",pos:C[n].position};R.drawCircle(t,u);let p=t.append("text").attr("visibility","hidden").text(n),f=p.node().getBoundingClientRect().width;p.remove();let g=[];if(f<=l)g=[n];else{let i=n.split(" "),a="";p=t.append("text").attr("visibility","hidden"),i.forEach(s=>{let c=a?`${a} ${s}`:s;if(p.text(c),p.node().getBoundingClientRect().width>l){if(a&&g.push(a),a=s,p.text(s),p.node().getBoundingClientRect().width>l){let d="";for(let o of s)d+=o,p.text(d+"-"),p.node().getBoundingClientRect().width>l&&(g.push(d.slice(0,-1)+"-"),d=o);a=d}}else a=c}),a&&g.push(a),p.remove()}g.forEach((i,a)=>{let s={x:40,y:y+7+a*20,fill:"#666",text:i,textMargin:e.boxTextMargin??5},c=R.drawText(t,s).node().getBoundingClientRect().width;c>z&&c>e.leftMargin-c&&(z=c)}),y+=Math.max(20,g.length*20)})}r(ct,"drawActorLegend");var M=j().journey,E=0,Yt=r(function(t,e,l,y){let n=j(),h=n.journey.titleColor,u=n.journey.titleFontSize,p=n.journey.titleFontFamily,f=n.securityLevel,g;f==="sandbox"&&(g=it("#i"+e));let i=it(f==="sandbox"?g.nodes()[0].contentDocument.body:"body");w.init();let a=i.select("#"+e);R.initGraphics(a,e);let s=y.db.getTasks(),c=y.db.getDiagramTitle(),d=y.db.getActors();for(let A in C)delete C[A];let o=0;d.forEach(A=>{C[A]={color:M.actorColours[o%M.actorColours.length],position:o},o++}),ct(a),E=M.leftMargin+z,w.insert(0,0,E,Object.keys(C).length*50),Wt(a,s,0,e);let k=w.getBounds();c&&a.append("text").text(c).attr("x",E).attr("font-size",u).attr("font-weight","bold").attr("y",25).attr("fill",h).attr("font-family",p);let m=k.stopy-k.starty+2*M.diagramMarginY,b=E+k.stopx+2*M.diagramMarginX;_t(a,m,b,M.useMaxWidth),a.append("line").attr("x1",E).attr("y1",M.height*4).attr("x2",b-E-4).attr("y2",M.height*4).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#"+e+"-arrowhead)");let V=c?70:0;a.attr("viewBox",`${k.startx} -25 ${b} ${m+V}`),a.attr("preserveAspectRatio","xMinYMin meet"),a.attr("height",m+V+25)},"draw"),w={data:{startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},verticalPos:0,sequenceItems:[],init:r(function(){this.sequenceItems=[],this.data={startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},this.verticalPos=0},"init"),updateVal:r(function(t,e,l,y){t[e]===void 0?t[e]=l:t[e]=y(l,t[e])},"updateVal"),updateBounds:r(function(t,e,l,y){let n=j().journey,h=this,u=0;function p(f){return r(function(g){u++;let i=h.sequenceItems.length-u+1;h.updateVal(g,"starty",e-i*n.boxMargin,Math.min),h.updateVal(g,"stopy",y+i*n.boxMargin,Math.max),h.updateVal(w.data,"startx",t-i*n.boxMargin,Math.min),h.updateVal(w.data,"stopx",l+i*n.boxMargin,Math.max),f!=="activation"&&(h.updateVal(g,"startx",t-i*n.boxMargin,Math.min),h.updateVal(g,"stopx",l+i*n.boxMargin,Math.max),h.updateVal(w.data,"starty",e-i*n.boxMargin,Math.min),h.updateVal(w.data,"stopy",y+i*n.boxMargin,Math.max))},"updateItemBounds")}r(p,"updateFn"),this.sequenceItems.forEach(p())},"updateBounds"),insert:r(function(t,e,l,y){let n=Math.min(t,l),h=Math.max(t,l),u=Math.min(e,y),p=Math.max(e,y);this.updateVal(w.data,"startx",n,Math.min),this.updateVal(w.data,"starty",u,Math.min),this.updateVal(w.data,"stopx",h,Math.max),this.updateVal(w.data,"stopy",p,Math.max),this.updateBounds(n,u,h,p)},"insert"),bumpVerticalPos:r(function(t){this.verticalPos+=t,this.data.stopy=this.verticalPos},"bumpVerticalPos"),getVerticalPos:r(function(){return this.verticalPos},"getVerticalPos"),getBounds:r(function(){return this.data},"getBounds")},K=M.sectionFills,ht=M.sectionColours,Wt=r(function(t,e,l,y){let n=j().journey,h="",u=l+(n.height*2+n.diagramMarginY),p=0,f="#CCC",g="black",i=0;for(let[a,s]of e.entries()){if(h!==s.section){f=K[p%K.length],i=p%K.length,g=ht[p%ht.length];let d=0,o=s.section;for(let m=a;m<e.length&&e[m].section==o;m++)d+=1;let k={x:a*n.taskMargin+a*n.width+E,y:50,text:s.section,fill:f,num:i,colour:g,taskCount:d};R.drawSection(t,k,n),h=s.section,p++}let c=s.people.reduce((d,o)=>(C[o]&&(d[o]=C[o]),d),{});s.x=a*n.taskMargin+a*n.width+E,s.y=u,s.width=n.diagramMarginX,s.height=n.diagramMarginY,s.colour=g,s.fill=f,s.num=i,s.actors=c,R.drawTask(t,s,n,y),w.insert(s.x,s.y,s.x+s.width+n.taskMargin,450)}},"drawTasks"),ut={setConf:Dt,draw:Yt},qt={parser:Et,db:rt,renderer:ut,styles:Bt,init:r(t=>{ut.setConf(t.journey),rt.clear()},"init")};export{qt as diagram};
