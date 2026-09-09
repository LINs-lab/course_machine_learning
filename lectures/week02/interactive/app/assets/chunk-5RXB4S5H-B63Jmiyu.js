var H;import{t as ee}from"./purify.es-Bwm8T4QC.js";import{_ as se,c as ie}from"./chunk-NSK5VX7P-bxC5WUDr.js";import{u as Tt}from"./src-DRYM6eUR.js";import{n as u}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{t as T}from"./src-Be2gSjc_.js";import{H as re,K as ne,U as oe,a as ae,s as K,v as le,w as ce,x as O,y as he}from"./chunk-I66GZJ75-CgJN5WrL.js";import{r as de}from"./chunk-J7OUQ5F2-DK5phbYO.js";import{t as ue}from"./chunk-2GRJ4B5K-B92Uwj7z.js";import{t as pe}from"./chunk-XXDRQBXY-D_KNH1wc.js";import{t as ye}from"./chunk-KBJHAD2P-BKxl1MSu.js";var Et=(function(){var t=u(function(o,h,g,S){for(g||(g={}),S=o.length;S--;g[o[S]]=h);return g},"o"),e=[1,2],s=[1,3],n=[1,4],i=[2,4],a=[1,9],c=[1,11],d=[1,16],y=[1,17],b=[1,18],m=[1,19],x=[1,33],L=[1,20],N=[1,21],$=[1,22],R=[1,23],D=[1,24],p=[1,26],v=[1,27],k=[1,28],F=[1,29],B=[1,30],P=[1,31],G=[1,32],rt=[1,35],nt=[1,36],ot=[1,37],at=[1,38],X=[1,34],f=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],lt=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],vt=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],ft={trace:u(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"CLICK",39:"STRING",40:"HREF",41:"classDef",42:"CLASSDEF_ID",43:"CLASSDEF_STYLEOPTS",44:"DEFAULT",45:"style",46:"STYLE_IDS",47:"STYLEDEF_STYLEOPTS",48:"class",49:"CLASSENTITY_IDS",50:"STYLECLASS",51:"direction_tb",52:"direction_bt",53:"direction_rl",54:"direction_lr",56:";",57:"EDGE_STATE",58:"STYLE_SEPARATOR",59:"left_of",60:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:u(function(o,h,g,S,_,r,j){var l=r.length-1;switch(_){case 3:return S.setRootDoc(r[l]),r[l];case 4:this.$=[];break;case 5:r[l]!="nl"&&(r[l-1].push(r[l]),this.$=r[l-1]);break;case 6:case 7:this.$=r[l];break;case 8:this.$="nl";break;case 12:this.$=r[l];break;case 13:let ht=r[l-1];ht.description=S.trimColon(r[l]),this.$=ht;break;case 14:this.$={stmt:"relation",state1:r[l-2],state2:r[l]};break;case 15:let dt=S.trimColon(r[l]);this.$={stmt:"relation",state1:r[l-3],state2:r[l-1],description:dt};break;case 19:this.$={stmt:"state",id:r[l-3],type:"default",description:"",doc:r[l-1]};break;case 20:var z=r[l],V=r[l-2].trim();if(r[l].match(":")){var Z=r[l].split(":");z=Z[0],V=[V,Z[1]]}this.$={stmt:"state",id:z,type:"default",description:V};break;case 21:this.$={stmt:"state",id:r[l-3],type:"default",description:r[l-5],doc:r[l-1]};break;case 22:this.$={stmt:"state",id:r[l],type:"fork"};break;case 23:this.$={stmt:"state",id:r[l],type:"join"};break;case 24:this.$={stmt:"state",id:r[l],type:"choice"};break;case 25:this.$={stmt:"state",id:S.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:r[l-1].trim(),note:{position:r[l-2].trim(),text:r[l].trim()}};break;case 29:this.$=r[l].trim(),S.setAccTitle(this.$);break;case 30:case 31:this.$=r[l].trim(),S.setAccDescription(this.$);break;case 32:this.$={stmt:"click",id:r[l-3],url:r[l-2],tooltip:r[l-1]};break;case 33:this.$={stmt:"click",id:r[l-3],url:r[l-1],tooltip:""};break;case 34:case 35:this.$={stmt:"classDef",id:r[l-1].trim(),classes:r[l].trim()};break;case 36:this.$={stmt:"style",id:r[l-1].trim(),styleClass:r[l].trim()};break;case 37:this.$={stmt:"applyClass",id:r[l-1].trim(),styleClass:r[l].trim()};break;case 38:S.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 39:S.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 40:S.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 41:S.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 44:case 45:this.$={stmt:"state",id:r[l].trim(),type:"default",description:""};break;case 46:this.$={stmt:"state",id:r[l-2].trim(),classes:[r[l].trim()],type:"default",description:""};break;case 47:this.$={stmt:"state",id:r[l-2].trim(),classes:[r[l].trim()],type:"default",description:""};break}},"anonymous"),table:[{3:1,4:e,5:s,6:n},{1:[3]},{3:5,4:e,5:s,6:n},{3:6,4:e,5:s,6:n},t([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],i,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:a,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:y,19:b,22:m,24:x,25:L,26:N,27:$,28:R,29:D,32:25,33:p,35:v,37:k,38:F,41:B,45:P,48:G,51:rt,52:nt,53:ot,54:at,57:X},t(f,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:d,17:y,19:b,22:m,24:x,25:L,26:N,27:$,28:R,29:D,32:25,33:p,35:v,37:k,38:F,41:B,45:P,48:G,51:rt,52:nt,53:ot,54:at,57:X},t(f,[2,7]),t(f,[2,8]),t(f,[2,9]),t(f,[2,10]),t(f,[2,11]),t(f,[2,12],{14:[1,40],15:[1,41]}),t(f,[2,16]),{18:[1,42]},t(f,[2,18],{20:[1,43]}),{23:[1,44]},t(f,[2,22]),t(f,[2,23]),t(f,[2,24]),t(f,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},t(f,[2,28]),{34:[1,49]},{36:[1,50]},t(f,[2,31]),{13:51,24:x,57:X},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},t(lt,[2,44],{58:[1,56]}),t(lt,[2,45],{58:[1,57]}),t(f,[2,38]),t(f,[2,39]),t(f,[2,40]),t(f,[2,41]),t(f,[2,6]),t(f,[2,13]),{13:58,24:x,57:X},t(f,[2,17]),t(vt,i,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},t(f,[2,29]),t(f,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},t(f,[2,14],{14:[1,71]}),{4:a,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:y,19:b,21:[1,72],22:m,24:x,25:L,26:N,27:$,28:R,29:D,32:25,33:p,35:v,37:k,38:F,41:B,45:P,48:G,51:rt,52:nt,53:ot,54:at,57:X},t(f,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},t(f,[2,34]),t(f,[2,35]),t(f,[2,36]),t(f,[2,37]),t(lt,[2,46]),t(lt,[2,47]),t(f,[2,15]),t(f,[2,19]),t(vt,i,{7:78}),t(f,[2,26]),t(f,[2,27]),{5:[1,79]},{5:[1,80]},{4:a,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:d,17:y,19:b,21:[1,81],22:m,24:x,25:L,26:N,27:$,28:R,29:D,32:25,33:p,35:v,37:k,38:F,41:B,45:P,48:G,51:rt,52:nt,53:ot,54:at,57:X},t(f,[2,32]),t(f,[2,33]),t(f,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:u(function(o,h){if(h.recoverable)this.trace(o);else{var g=Error(o);throw g.hash=h,g}},"parseError"),parse:u(function(o){var h=this,g=[0],S=[],_=[null],r=[],j=this.table,l="",z=0,V=0,Z=0,ht=2,dt=1,Qt=r.slice.call(arguments,1),E=Object.create(this.lexer),M={yy:{}};for(var St in this.yy)Object.prototype.hasOwnProperty.call(this.yy,St)&&(M.yy[St]=this.yy[St]);E.setInput(o,M.yy),M.yy.lexer=E,M.yy.parser=this,E.yylloc===void 0&&(E.yylloc={});var bt=E.yylloc;r.push(bt);var Zt=E.options&&E.options.ranges;typeof M.yy.parseError=="function"?this.parseError=M.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function te(w){g.length-=2*w,_.length-=w,r.length-=w}u(te,"popStack");function It(){var w=S.pop()||E.lex()||dt;return typeof w!="number"&&(w instanceof Array&&(S=w,w=S.pop()),w=h.symbols_[w]||w),w}u(It,"lex");for(var I,_t,U,A,kt,J={},ut,Y,Lt,pt;;){if(U=g[g.length-1],this.defaultActions[U]?A=this.defaultActions[U]:(I??(I=It()),A=j[U]&&j[U][I]),A===void 0||!A.length||!A[0]){var At="";for(ut in pt=[],j[U])this.terminals_[ut]&&ut>ht&&pt.push("'"+this.terminals_[ut]+"'");At=E.showPosition?"Parse error on line "+(z+1)+`:
`+E.showPosition()+`
Expecting `+pt.join(", ")+", got '"+(this.terminals_[I]||I)+"'":"Parse error on line "+(z+1)+": Unexpected "+(I==dt?"end of input":"'"+(this.terminals_[I]||I)+"'"),this.parseError(At,{text:E.match,token:this.terminals_[I]||I,line:E.yylineno,loc:bt,expected:pt})}if(A[0]instanceof Array&&A.length>1)throw Error("Parse Error: multiple actions possible at state: "+U+", token: "+I);switch(A[0]){case 1:g.push(I),_.push(E.yytext),r.push(E.yylloc),g.push(A[1]),I=null,_t?(I=_t,_t=null):(V=E.yyleng,l=E.yytext,z=E.yylineno,bt=E.yylloc,Z>0&&Z--);break;case 2:if(Y=this.productions_[A[1]][1],J.$=_[_.length-Y],J._$={first_line:r[r.length-(Y||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(Y||1)].first_column,last_column:r[r.length-1].last_column},Zt&&(J._$.range=[r[r.length-(Y||1)].range[0],r[r.length-1].range[1]]),kt=this.performAction.apply(J,[l,V,z,M.yy,A[1],_,r].concat(Qt)),kt!==void 0)return kt;Y&&(g=g.slice(0,-1*Y*2),_=_.slice(0,-1*Y),r=r.slice(0,-1*Y)),g.push(this.productions_[A[1]][0]),_.push(J.$),r.push(J._$),Lt=j[g[g.length-2]][g[g.length-1]],g.push(Lt);break;case 3:return!0}}return!0},"parse")};ft.lexer=(function(){return{EOF:1,parseError:u(function(o,h){if(this.yy.parser)this.yy.parser.parseError(o,h);else throw Error(o)},"parseError"),setInput:u(function(o,h){return this.yy=h||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:u(function(){var o=this._input[0];return this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o,o.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:u(function(o){var h=o.length,g=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-h),this.offset-=h;var S=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),g.length-1&&(this.yylineno-=g.length-1);var _=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:g?(g.length===S.length?this.yylloc.first_column:0)+S[S.length-g.length].length-g[0].length:this.yylloc.first_column-h},this.options.ranges&&(this.yylloc.range=[_[0],_[0]+this.yyleng-h]),this.yyleng=this.yytext.length,this},"unput"),more:u(function(){return this._more=!0,this},"more"),reject:u(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:u(function(o){this.unput(this.match.slice(o))},"less"),pastInput:u(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:u(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:u(function(){var o=this.pastInput(),h=Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+h+"^"},"showPosition"),test_match:u(function(o,h){var g,S,_;if(this.options.backtrack_lexer&&(_={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(_.yylloc.range=this.yylloc.range.slice(0))),S=o[0].match(/(?:\r\n?|\n).*/g),S&&(this.yylineno+=S.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:S?S[S.length-1].length-S[S.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],g=this.performAction.call(this,this.yy,this,h,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),g)return g;if(this._backtrack){for(var r in _)this[r]=_[r];return!1}return!1},"test_match"),next:u(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,h,g,S;this._more||(this.yytext="",this.match="");for(var _=this._currentRules(),r=0;r<_.length;r++)if(g=this._input.match(this.rules[_[r]]),g&&(!h||g[0].length>h[0].length)){if(h=g,S=r,this.options.backtrack_lexer){if(o=this.test_match(g,_[r]),o!==!1)return o;if(this._backtrack){h=!1;continue}else return!1}else if(!this.options.flex)break}return h?(o=this.test_match(h,_[S]),o===!1?!1:o):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:u(function(){return this.next()||this.lex()},"lex"),begin:u(function(o){this.conditionStack.push(o)},"begin"),popState:u(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:u(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:u(function(o){return o=this.conditionStack.length-1-Math.abs(o||0),o>=0?this.conditionStack[o]:"INITIAL"},"topState"),pushState:u(function(o){this.begin(o)},"pushState"),stateStackSize:u(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:u(function(o,h,g,S){function _(){let r=h.yytext.indexOf("%%");if(r===0)return!1;if(r>0){let j=h.yytext.slice(0,r),l=h.yytext.slice(r);l&&o.lexer.unput(l),h.yytext=j}return!0}switch(u(_,"processId"),g){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:return 51;case 5:return 52;case 6:return 53;case 7:return 54;case 8:return 5;case 9:break;case 10:break;case 11:break;case 12:break;case 13:return this.pushState("SCALE"),17;case 14:return 18;case 15:this.popState();break;case 16:return this.begin("acc_title"),33;case 17:return this.popState(),"acc_title_value";case 18:return this.begin("acc_descr"),35;case 19:return this.popState(),"acc_descr_value";case 20:this.begin("acc_descr_multiline");break;case 21:this.popState();break;case 22:return"acc_descr_multiline_value";case 23:return this.pushState("CLASSDEF"),41;case 24:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";case 25:return this.popState(),this.pushState("CLASSDEFID"),42;case 26:return this.popState(),43;case 27:return this.pushState("CLASS"),48;case 28:return this.popState(),this.pushState("CLASS_STYLE"),49;case 29:return this.popState(),50;case 30:return this.pushState("STYLE"),45;case 31:return this.popState(),this.pushState("STYLEDEF_STYLES"),46;case 32:return this.popState(),47;case 33:return this.pushState("SCALE"),17;case 34:return 18;case 35:this.popState();break;case 36:this.pushState("STATE");break;case 37:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),25;case 38:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),26;case 39:return this.popState(),h.yytext=h.yytext.slice(0,-10).trim(),27;case 40:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),25;case 41:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),26;case 42:return this.popState(),h.yytext=h.yytext.slice(0,-10).trim(),27;case 43:return 51;case 44:return 52;case 45:return 53;case 46:return 54;case 47:this.pushState("STATE_STRING");break;case 48:return this.pushState("STATE_ID"),"AS";case 49:return _()?(this.popState(),"ID"):void 0;case 50:this.popState();break;case 51:return"STATE_DESCR";case 52:throw Error('Error: State name must be a single word. Found: "'+h.yytext.trim()+'"');case 53:return 19;case 54:this.popState();break;case 55:return this.popState(),this.pushState("struct"),20;case 56:return this.popState(),21;case 57:break;case 58:return this.begin("NOTE"),29;case 59:return this.popState(),this.pushState("NOTE_ID"),59;case 60:return this.popState(),this.pushState("NOTE_ID"),60;case 61:this.popState(),this.pushState("FLOATING_NOTE");break;case 62:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";case 63:break;case 64:return"NOTE_TEXT";case 65:return _()?(this.popState(),"ID"):void 0;case 66:return _()?(this.popState(),this.pushState("NOTE_TEXT"),24):void 0;case 67:return this.popState(),h.yytext=h.yytext.substr(2).trim(),31;case 68:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),31;case 69:return 6;case 70:return 6;case 71:return 16;case 72:return 57;case 73:return _()?24:void 0;case 74:return h.yytext=h.yytext.trim(),14;case 75:return 15;case 76:return 28;case 77:return 58;case 78:return 5;case 79:return"INVALID"}},"anonymous"),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:\w+\s+\w+.*?\{)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?\n\s*end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:(?:[^:\n;]|:[^:\n;])+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[10,11,12],inclusive:!1},struct:{rules:[10,11,12,23,27,30,36,43,44,45,46,56,57,58,72,73,74,75,76,77],inclusive:!1},FLOATING_NOTE_ID:{rules:[65],inclusive:!1},FLOATING_NOTE:{rules:[62,63,64],inclusive:!1},NOTE_TEXT:{rules:[67,68],inclusive:!1},NOTE_ID:{rules:[66],inclusive:!1},NOTE:{rules:[59,60,61],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[32],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[31],inclusive:!1},CLASS_STYLE:{rules:[29],inclusive:!1},CLASS:{rules:[28],inclusive:!1},CLASSDEFID:{rules:[26],inclusive:!1},CLASSDEF:{rules:[24,25],inclusive:!1},acc_descr_multiline:{rules:[21,22],inclusive:!1},acc_descr:{rules:[19],inclusive:!1},acc_title:{rules:[17],inclusive:!1},SCALE:{rules:[14,15,34,35],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[49],inclusive:!1},STATE_STRING:{rules:[50,51],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[10,11,12,37,38,39,40,41,42,47,48,52,53,54,55],inclusive:!1},ID:{rules:[10,11,12],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,11,12,13,16,18,20,23,27,30,33,36,55,58,69,70,71,72,73,74,75,77,78,79],inclusive:!0}}}})();function ct(){this.yy={}}return u(ct,"Parser"),ct.prototype=ft,ft.Parser=ct,new ct})();Et.parser=Et;var ge=Et,me="TB",wt="TB",Nt="dir",q="state",Q="root",xt="relation",fe="classDef",Se="style",be="applyClass",tt="default",Ot="divider",Rt="fill:none",Bt="fill: #333",Ft="c",Yt="markdown",Pt="normal",Dt="rect",Ct="rectWithTitle",_e="stateStart",ke="stateEnd",Gt="divider",Wt="roundedWithTitle",Te="note",Ee="noteGroup",et="statediagram",xe=`${et}-state`,jt="transition",De="note",Ce=`${jt} note-edge`,$e=`${et}-${De}`,ve=`${et}-cluster`,Ie=`${et}-cluster-alt`,zt="parent",Mt="note",Le="state",$t="----",Ae=`${$t}${Mt}`,Ut=`${$t}${zt}`,Kt=u((t,e=wt)=>{if(!t.doc)return e;let s=e;for(let n of t.doc)n.stmt==="dir"&&(s=n.value);return s},"getDir"),we={getClasses:u(function(t,e){return e.db.getClasses()},"getClasses"),draw:u(async function(t,e,s,n){T.info("REF0:"),T.info("Drawing state diagram (v2)",e);let{securityLevel:i,state:a,layout:c}=O();n.db.extract(n.db.getRootDocV2());let d=n.db.getData(),y=pe(e,i);d.type=n.type,d.layoutAlgorithm=c,d.nodeSpacing=(a==null?void 0:a.nodeSpacing)||50,d.rankSpacing=(a==null?void 0:a.rankSpacing)||50,O().look==="neo"?d.markers=["barbNeo"]:d.markers=["barb"],d.diagramId=e,await de(d,y);try{(typeof n.db.getLinks=="function"?n.db.getLinks():new Map).forEach((b,m)=>{var v;let x=typeof m=="string"?m:typeof(m==null?void 0:m.id)=="string"?m.id:"",L=d.nodes.find(k=>k.id===x);if(!x){T.warn("\u26A0\uFE0F Invalid or missing stateId from key:",JSON.stringify(m));return}let N=(v=y.node())==null?void 0:v.querySelectorAll("g.node, g.rough-node"),$;if(N==null||N.forEach(k=>{var B;let F=(B=k.textContent)==null?void 0:B.trim();(k.id===(L==null?void 0:L.domId)||F===x)&&($=k)}),!$){T.warn("\u26A0\uFE0F Could not find node matching text:",x);return}let R=$.parentNode;if(!R){T.warn("\u26A0\uFE0F Node has no parent, cannot wrap:",x);return}let D=document.createElementNS("http://www.w3.org/2000/svg","a"),p=b.url.replace(/^"+|"+$/g,"");if(D.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",p),D.setAttribute("target","_blank"),b.tooltip){let k=b.tooltip.replace(/^"+|"+$/g,"");D.setAttribute("title",k),$.setAttribute("title",k)}R.replaceChild(D,$),D.appendChild($),T.info("\u{1F517} Wrapped node in <a> tag for:",x,b.url)})}catch(b){T.error("\u274C Error injecting clickable links:",b)}se.insertTitle(y,"statediagramTitleText",(a==null?void 0:a.titleTopMargin)??25,n.db.getDiagramTitle()),ye(y,8,et,(a==null?void 0:a.useMaxWidth)??!0)},"draw"),getDir:Kt},yt=new Map,W=0;function gt(t="",e=0,s="",n=$t){return`${Le}-${t}${s!==null&&s.length>0?`${n}${s}`:""}-${e}`}u(gt,"stateDomId");var Ne=u((t,e,s,n,i,a,c,d)=>{T.trace("items",e),e.forEach(y=>{switch(y.stmt){case q:it(t,y,s,n,i,a,c,d);break;case tt:it(t,y,s,n,i,a,c,d);break;case xt:{it(t,y.state1,s,n,i,a,c,d),it(t,y.state2,s,n,i,a,c,d);let b=c==="neo",m={id:"edge"+W,start:y.state1.id,end:y.state2.id,arrowhead:"normal",arrowTypeEnd:b?"arrow_barb_neo":"arrow_barb",style:Rt,labelStyle:"",label:K.sanitizeText(y.description??"",O()),arrowheadStyle:Bt,labelpos:Ft,labelType:Yt,thickness:Pt,classes:jt,look:c};i.push(m),W++}break}})},"setupDoc"),Ht=u((t,e=wt)=>{let s=e;if(t.doc)for(let n of t.doc)n.stmt==="dir"&&(s=n.value);return s},"getDir");function st(t,e,s){if(!e.id||e.id==="</join></fork>"||e.id==="</choice>")return;e.cssClasses&&(Array.isArray(e.cssCompiledStyles)||(e.cssCompiledStyles=[]),e.cssClasses.split(" ").forEach(i=>{let a=s.get(i);a&&(e.cssCompiledStyles=[...e.cssCompiledStyles??[],...a.styles])}));let n=t.find(i=>i.id===e.id);n?Object.assign(n,e):t.push(e)}u(st,"insertOrUpdateNode");function Xt(t){var e;return((e=t==null?void 0:t.classes)==null?void 0:e.join(" "))??""}u(Xt,"getClassesFromDbInfo");function Vt(t){return(t==null?void 0:t.styles)??[]}u(Vt,"getStylesFromDbInfo");var it=u((t,e,s,n,i,a,c,d)=>{var N,$,R;let y=e.id,b=s.get(y),m=Xt(b),x=Vt(b),L=O();if(T.info("dataFetcher parsedItem",e,b,x),y!=="root"){let D=Dt;e.start===!0?D=_e:e.start===!1&&(D=ke),e.type!==tt&&(D=e.type),yt.get(y)||yt.set(y,{id:y,shape:D,description:K.sanitizeText(y,L),cssClasses:`${m} ${xe}`,cssStyles:x});let p=yt.get(y);e.description&&(Array.isArray(p.description)?(p.shape=Ct,p.description.push(e.description)):(N=p.description)!=null&&N.length&&p.description.length>0?(p.shape=Ct,p.description===y?p.description=[e.description]:p.description=[p.description,e.description]):(p.shape=Dt,p.description=e.description),p.description=K.sanitizeTextOrArray(p.description,L)),(($=p.description)==null?void 0:$.length)===1&&p.shape===Ct&&(p.type==="group"?p.shape=Wt:p.shape=Dt),!p.type&&e.doc&&(T.info("Setting cluster for XCX",y,Ht(e)),p.type="group",p.isGroup=!0,p.dir=Ht(e),p.explicitDir=e.doc.some(k=>k.stmt==="dir"),p.shape=e.type===Ot?Gt:Wt,p.cssClasses=`${p.cssClasses} ${ve} ${a?Ie:""}`);let v={labelStyle:"",shape:p.shape,label:p.description,cssClasses:p.cssClasses,cssCompiledStyles:[],cssStyles:p.cssStyles,id:y,dir:p.dir,domId:gt(y,W),type:p.type,isGroup:p.type==="group",padding:8,rx:10,ry:10,look:c,labelType:"markdown"};if(v.shape===Gt&&(v.label=""),t&&t.id!=="root"&&(T.trace("Setting node ",y," to be child of its parent ",t.id),v.parentId=t.id),v.centerLabel=!0,e.note){let k={labelStyle:"",shape:Te,label:e.note.text,labelType:"markdown",cssClasses:$e,cssStyles:[],cssCompiledStyles:[],id:y+Ae+"-"+W,domId:gt(y,W,Mt),type:p.type,isGroup:p.type==="group",padding:(R=L.flowchart)==null?void 0:R.padding,look:c,position:e.note.position},F=y+Ut,B={labelStyle:"",shape:Ee,label:e.note.text,cssClasses:p.cssClasses,cssStyles:[],id:y+Ut,domId:gt(y,W,zt),type:"group",isGroup:!0,padding:16,look:c,position:e.note.position};W++,B.id=F,k.parentId=F,st(n,B,d),st(n,k,d),st(n,v,d);let P=y,G=k.id;e.note.position==="left of"&&(P=k.id,G=y),i.push({id:P+"-"+G,start:P,end:G,arrowhead:"none",arrowTypeEnd:"",style:Rt,labelStyle:"",classes:Ce,arrowheadStyle:Bt,labelpos:Ft,labelType:Yt,thickness:Pt,look:c})}else st(n,v,d)}e.doc&&(T.trace("Adding nodes children "),Ne(e,e.doc,s,n,i,!a,c,d))},"dataFetcher"),Oe=u(()=>{yt.clear(),W=0},"reset"),C={START_NODE:"[*]",START_TYPE:"start",END_NODE:"[*]",END_TYPE:"end",COLOR_KEYWORD:"color",FILL_KEYWORD:"fill",BG_FILL:"bgFill",STYLECLASS_SEP:","},Jt=u(()=>new Map,"newClassesList"),qt=u(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),mt=u(t=>JSON.parse(JSON.stringify(t)),"clone"),Re=(H=class{constructor(e){this.version=e,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=Jt(),this.documents={root:qt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.funs=[],this.getAccTitle=he,this.setAccTitle=oe,this.getAccDescription=le,this.setAccDescription=re,this.setDiagramTitle=ne,this.getDiagramTitle=ce,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this),this.bindFunctions=this.bindFunctions.bind(this)}extract(e){this.clear(!0);for(let i of Array.isArray(e)?e:e.doc)switch(i.stmt){case q:this.addState(i.id.trim(),i.type,i.doc,i.description,i.note);break;case xt:this.addRelation(i.state1,i.state2,i.description);break;case fe:this.addStyleClass(i.id.trim(),i.classes);break;case Se:this.handleStyleDef(i);break;case be:this.setCssClass(i.id.trim(),i.styleClass);break;case"click":this.addLink(i.id,i.url,i.tooltip);break}let s=this.getStates(),n=O();Oe(),it(void 0,this.getRootDocV2(),s,this.nodes,this.edges,!0,n.look,this.classes);for(let i of this.nodes)if(Array.isArray(i.label)){if(i.description=i.label.slice(1),i.isGroup&&i.description.length>0)throw Error(`Group nodes can only have label. Remove the additional description for node [${i.id}]`);i.label=i.label[0]}}handleStyleDef(e){let s=e.id.trim().split(","),n=e.styleClass.split(",");for(let i of s){let a=this.getState(i);if(!a){let c=i.trim();this.addState(c),a=this.getState(c)}a&&(a.styles=n.map(c=>{var d;return(d=c.replace(/;/g,""))==null?void 0:d.trim()}))}}setRootDoc(e){T.info("Setting root doc",e),this.rootDoc=e,this.version===1?this.extract(e):this.extract(this.getRootDocV2())}docTranslator(e,s,n){if(s.stmt===xt){this.docTranslator(e,s.state1,!0),this.docTranslator(e,s.state2,!1);return}if(s.stmt===q&&(s.id===C.START_NODE?(s.id=e.id+(n?"_start":"_end"),s.start=n):s.id=s.id.trim()),s.stmt!==Q&&s.stmt!==q||!s.doc)return;let i=[],a=[];for(let c of s.doc)if(c.type===Ot){let d=mt(c);d.doc=mt(a),i.push(d),a=[]}else a.push(c);if(i.length>0&&a.length>0){let c={stmt:q,id:ie(),type:"divider",doc:mt(a)};i.push(mt(c)),s.doc=i}s.doc.forEach(c=>this.docTranslator(s,c,!0))}getRootDocV2(){return this.docTranslator({id:Q,stmt:Q},{id:Q,stmt:Q,doc:this.rootDoc},!0),{id:Q,doc:this.rootDoc}}addState(e,s=tt,n=void 0,i=void 0,a=void 0,c=void 0,d=void 0,y=void 0){let b=e==null?void 0:e.trim();if(!this.currentDocument.states.has(b))T.info("Adding state ",b,i),this.currentDocument.states.set(b,{stmt:q,id:b,descriptions:[],type:s,doc:n,note:a,classes:[],styles:[],textStyles:[]});else{let m=this.currentDocument.states.get(b);if(!m)throw Error(`State not found: ${b}`);m.doc||(m.doc=n),m.type||(m.type=s)}if(i&&(T.info("Setting state description",b,i),(Array.isArray(i)?i:[i]).forEach(m=>this.addDescription(b,m.trim()))),a){let m=this.currentDocument.states.get(b);if(!m)throw Error(`State not found: ${b}`);m.note=a,m.note.text=K.sanitizeText(m.note.text,O())}c&&(T.info("Setting state classes",b,c),(Array.isArray(c)?c:[c]).forEach(m=>this.setCssClass(b,m.trim()))),d&&(T.info("Setting state styles",b,d),(Array.isArray(d)?d:[d]).forEach(m=>this.setStyle(b,m.trim()))),y&&(T.info("Setting state styles",b,d),(Array.isArray(y)?y:[y]).forEach(m=>this.setTextStyle(b,m.trim())))}clear(e){this.nodes=[],this.edges=[],this.funs=[this.setupToolTips.bind(this)],this.documents={root:qt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=Jt(),e||(this.links=new Map,ae())}getState(e){return this.currentDocument.states.get(e)}getStates(){return this.currentDocument.states}logDocuments(){T.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}addLink(e,s,n){this.links.set(e,{url:s,tooltip:n}),T.warn("Adding link",e,s,n)}getLinks(){return this.links}startIdIfNeeded(e=""){return e===C.START_NODE?(this.startEndCount++,`${C.START_TYPE}${this.startEndCount}`):e}startTypeIfNeeded(e="",s=tt){return e===C.START_NODE?C.START_TYPE:s}endIdIfNeeded(e=""){return e===C.END_NODE?(this.startEndCount++,`${C.END_TYPE}${this.startEndCount}`):e}endTypeIfNeeded(e="",s=tt){return e===C.END_NODE?C.END_TYPE:s}addRelationObjs(e,s,n=""){let i=this.startIdIfNeeded(e.id.trim()),a=this.startTypeIfNeeded(e.id.trim(),e.type),c=this.startIdIfNeeded(s.id.trim()),d=this.startTypeIfNeeded(s.id.trim(),s.type);this.addState(i,a,e.doc,e.description,e.note,e.classes,e.styles,e.textStyles),this.addState(c,d,s.doc,s.description,s.note,s.classes,s.styles,s.textStyles),this.currentDocument.relations.push({id1:i,id2:c,relationTitle:K.sanitizeText(n,O())})}addRelation(e,s,n){if(typeof e=="object"&&typeof s=="object")this.addRelationObjs(e,s,n);else if(typeof e=="string"&&typeof s=="string"){let i=this.startIdIfNeeded(e.trim()),a=this.startTypeIfNeeded(e),c=this.endIdIfNeeded(s.trim()),d=this.endTypeIfNeeded(s);this.addState(i,a),this.addState(c,d),this.currentDocument.relations.push({id1:i,id2:c,relationTitle:n?K.sanitizeText(n,O()):void 0})}}addDescription(e,s){var a;let n=this.currentDocument.states.get(e),i=s.startsWith(":")?s.replace(":","").trim():s;(a=n==null?void 0:n.descriptions)==null||a.push(K.sanitizeText(i,O()))}cleanupLabel(e){return e.startsWith(":")?e.slice(2).trim():e.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(e,s=""){this.classes.has(e)||this.classes.set(e,{id:e,styles:[],textStyles:[]});let n=this.classes.get(e);s&&n&&s.split(C.STYLECLASS_SEP).forEach(i=>{let a=i.replace(/([^;]*);/,"$1").trim();if(RegExp(C.COLOR_KEYWORD).exec(i)){let c=a.replace(C.FILL_KEYWORD,C.BG_FILL).replace(C.COLOR_KEYWORD,C.FILL_KEYWORD);n.textStyles.push(c)}n.styles.push(a)})}getClasses(){return this.classes}setupToolTips(e){let s=ue();Tt(e).select("svg").selectAll("g.node, g.rough-node").on("mouseover",n=>{var d;let i=Tt(n.currentTarget),a=i.attr("title");if(a===null)return;let c=(d=n.currentTarget)==null?void 0:d.getBoundingClientRect();s.transition().duration(200).style("opacity",".9"),s.style("left",window.scrollX+c.left+(c.right-c.left)/2+"px").style("top",window.scrollY+c.bottom+"px"),s.html(ee.sanitize(a)),i.classed("hover",!0)}).on("mouseout",n=>{s.transition().duration(500).style("opacity",0),Tt(n.currentTarget).classed("hover",!1)})}setCssClass(e,s){e.split(",").forEach(n=>{var a;let i=this.getState(n);if(!i){let c=n.trim();this.addState(c),i=this.getState(c)}(a=i==null?void 0:i.classes)==null||a.push(s)})}setStyle(e,s){var n,i;(i=(n=this.getState(e))==null?void 0:n.styles)==null||i.push(s)}setTextStyle(e,s){var n,i;(i=(n=this.getState(e))==null?void 0:n.textStyles)==null||i.push(s)}bindFunctions(e){this.funs.forEach(s=>{s(e)})}getDirectionStatement(){return this.rootDoc.find(e=>e.stmt===Nt)}getDirection(){var e;return((e=this.getDirectionStatement())==null?void 0:e.value)??me}setDirection(e){let s=this.getDirectionStatement();s?s.value=e:this.rootDoc.unshift({stmt:Nt,value:e})}trimColon(e){return e.startsWith(":")?e.slice(1).trim():e.trim()}getData(){let e=O();return{nodes:this.nodes,edges:this.edges,other:{},config:e,direction:Kt(this.getRootDocV2())}}getConfig(){return O().state}},u(H,"StateDB"),H.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3},H),Be=u(t=>`
defs [id$="-barbEnd"] {
    fill: ${t.transitionColor};
    stroke: ${t.transitionColor};
  }
g.stateGroup text {
  fill: ${t.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${t.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${t.stateLabelColor};
}

g.stateGroup rect {
  fill: ${t.mainBkg};
  stroke: ${t.nodeBorder};
}

g.stateGroup line {
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.transition {
  stroke: ${t.transitionColor};
  stroke-width: ${t.strokeWidth||1};
  fill: none;
}

.stateGroup .composit {
  fill: ${t.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${t.noteBorderColor};
  fill: ${t.noteBkgColor};

  text {
    fill: ${t.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${t.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${t.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${t.edgeLabelBackground};
  p {
    background-color: ${t.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${t.edgeLabelBackground};
    fill: ${t.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${t.transitionLabelColor||t.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${t.transitionLabelColor||t.tertiaryTextColor};
}

.stateLabel text {
  fill: ${t.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node .fork-join {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node circle.state-end {
  fill: ${t.innerEndBackground};
  stroke: ${t.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${t.compositeBackground||t.background};
  // stroke: ${t.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${t.stateBkg||t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}
.node polygon {
  fill: ${t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};;
  stroke-width: ${t.strokeWidth||1}px;
}
[id$="-barbEnd"] {
  fill: ${t.lineColor};
}

.statediagram-cluster rect {
  fill: ${t.compositeTitleBackground};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}

.cluster-label, .nodeLabel {
  color: ${t.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${t.stateBorder||t.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${t.compositeBackground||t.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${t.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${t.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${t.noteTextColor};
}

[id$="-dependencyStart"], [id$="-dependencyEnd"] {
  fill: ${t.lineColor};
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${t.textColor};
}

[data-look="neo"].statediagram-cluster rect {
  fill: ${t.mainBkg};
  stroke: ${t.useGradient?"url("+t.svgId+"-gradient)":t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth??1};
}
[data-look="neo"].statediagram-cluster rect.outer {
  rx: ${t.radius}px;
  ry: ${t.radius}px;
  filter: ${t.dropShadow?t.dropShadow.replace("url(#drop-shadow)",`url(${t.svgId}-drop-shadow)`):"none"}
}
`,"getStyles");export{Be as i,ge as n,we as r,Re as t};
