import{o as lt,t as Tt}from"./chunk-C4rtOYze.js";import"./purify.es-Bwm8T4QC.js";import{t as ke}from"./linear-BmfCipCt.js";import{n as pe,o as ge,s as be}from"./time-BcZKVgVI.js";import"./defaultLocale-CGJY8D9C.js";import{C as Zt,N as qt,T as Xt,c as Qt,d as ve,f as Te,g as xe,h as $e,m as we,p as _e,t as Jt,u as De,v as Kt,x as te}from"./defaultLocale-DzliDDTm.js";import{_ as Se}from"./chunk-NSK5VX7P-bxC5WUDr.js";import{s as Me}from"./timer-BqJeK64m.js";import{u as St}from"./src-DRYM6eUR.js";import{n as h}from"./chunk-Y2CYZVJY-Ci6ix4_L.js";import{r as ee,t as at}from"./src-Be2gSjc_.js";import{H as Ce,K as Ee,U as Ye,a as Oe,c as Ae,s as Le,v as Ie,w as Fe,x as dt,y as We}from"./chunk-I66GZJ75-CgJN5WrL.js";import{t as Pe}from"./dist-BY5C0xw-.js";function He(t){return t}var xt=1,Mt=2,Ct=3,$t=4,ie=1e-6;function ze(t){return"translate("+t+",0)"}function Ne(t){return"translate(0,"+t+")"}function Be(t){return e=>+t(e)}function je(t,e){return e=Math.max(0,t.bandwidth()-e*2)/2,t.round()&&(e=Math.round(e)),r=>+t(r)+e}function Re(){return!this.__axis}function se(t,e){var r=[],i=null,a=null,f=6,g=6,w=3,E=typeof window<"u"&&window.devicePixelRatio>1?0:.5,_=t===xt||t===$t?-1:1,C=t===$t||t===Mt?"x":"y",P=t===xt||t===Ct?ze:Ne;function T($){var R=i??(e.ticks?e.ticks.apply(e,r):e.domain()),N=a??(e.tickFormat?e.tickFormat.apply(e,r):He),k=Math.max(f,0)+w,S=e.range(),A=+S[0]+E,F=+S[S.length-1]+E,B=(e.bandwidth?je:Be)(e.copy(),E),z=$.selection?$.selection():$,Y=z.selectAll(".domain").data([null]),b=z.selectAll(".tick").data(R,e).order(),y=b.exit(),M=b.enter().append("g").attr("class","tick"),m=b.select("line"),p=b.select("text");Y=Y.merge(Y.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor")),b=b.merge(M),m=m.merge(M.append("line").attr("stroke","currentColor").attr(C+"2",_*f)),p=p.merge(M.append("text").attr("fill","currentColor").attr(C,_*k).attr("dy",t===xt?"0em":t===Ct?"0.71em":"0.32em")),$!==z&&(Y=Y.transition($),b=b.transition($),m=m.transition($),p=p.transition($),y=y.transition($).attr("opacity",ie).attr("transform",function(s){return isFinite(s=B(s))?P(s+E):this.getAttribute("transform")}),M.attr("opacity",ie).attr("transform",function(s){var u=this.parentNode.__axis;return P((u&&isFinite(u=u(s))?u:B(s))+E)})),y.remove(),Y.attr("d",t===$t||t===Mt?g?"M"+_*g+","+A+"H"+E+"V"+F+"H"+_*g:"M"+E+","+A+"V"+F:g?"M"+A+","+_*g+"V"+E+"H"+F+"V"+_*g:"M"+A+","+E+"H"+F),b.attr("opacity",1).attr("transform",function(s){return P(B(s)+E)}),m.attr(C+"2",_*f),p.attr(C,_*k).text(N),z.filter(Re).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===Mt?"start":t===$t?"end":"middle"),z.each(function(){this.__axis=B})}return T.scale=function($){return arguments.length?(e=$,T):e},T.ticks=function(){return r=Array.from(arguments),T},T.tickArguments=function($){return arguments.length?(r=$==null?[]:Array.from($),T):r.slice()},T.tickValues=function($){return arguments.length?(i=$==null?null:Array.from($),T):i&&i.slice()},T.tickFormat=function($){return arguments.length?(a=$,T):a},T.tickSize=function($){return arguments.length?(f=g=+$,T):f},T.tickSizeInner=function($){return arguments.length?(f=+$,T):f},T.tickSizeOuter=function($){return arguments.length?(g=+$,T):g},T.tickPadding=function($){return arguments.length?(w=+$,T):w},T.offset=function($){return arguments.length?(E=+$,T):E},T}function Ge(t){return se(xt,t)}function Ve(t){return se(Ct,t)}var Ue=Tt(((t,e)=>{(function(r,i){typeof t=="object"&&e!==void 0?e.exports=i():typeof define=="function"&&define.amd?define(i):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_isoWeek=i()})(t,(function(){var r="day";return function(i,a,f){var g=function(_){return _.add(4-_.isoWeekday(),r)},w=a.prototype;w.isoWeekYear=function(){return g(this).year()},w.isoWeek=function(_){if(!this.$utils().u(_))return this.add(7*(_-this.isoWeek()),r);var C,P,T,$,R=g(this),N=(C=this.isoWeekYear(),P=this.$u,T=(P?f.utc:f)().year(C).startOf("year"),$=4-T.isoWeekday(),T.isoWeekday()>4&&($+=7),T.add($,r));return R.diff(N,"week")+1},w.isoWeekday=function(_){return this.$utils().u(_)?this.day()||7:this.day(this.day()%7?_:_-7)};var E=w.startOf;w.startOf=function(_,C){var P=this.$utils(),T=!!P.u(C)||C;return P.p(_)==="isoweek"?T?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):E.bind(this)(_,C)}}}))})),Ze=Tt(((t,e)=>{(function(r,i){typeof t=="object"&&e!==void 0?e.exports=i():typeof define=="function"&&define.amd?define(i):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_customParseFormat=i()})(t,(function(){var r={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},i=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,a=/\d/,f=/\d\d/,g=/\d\d?/,w=/\d*[^-_:/,()\s\d]+/,E={},_=function(k){return(k=+k)+(k>68?1900:2e3)},C=function(k){return function(S){this[k]=+S}},P=[/[+-]\d\d:?(\d\d)?|Z/,function(k){(this.zone||(this.zone={})).offset=(function(S){if(!S||S==="Z")return 0;var A=S.match(/([+-]|\d\d)/g),F=60*A[1]+(+A[2]||0);return F===0?0:A[0]==="+"?-F:F})(k)}],T=function(k){var S=E[k];return S&&(S.indexOf?S:S.s.concat(S.f))},$=function(k,S){var A,F=E.meridiem;if(F){for(var B=1;B<=24;B+=1)if(k.indexOf(F(B,0,S))>-1){A=B>12;break}}else A=k===(S?"pm":"PM");return A},R={A:[w,function(k){this.afternoon=$(k,!1)}],a:[w,function(k){this.afternoon=$(k,!0)}],Q:[a,function(k){this.month=3*(k-1)+1}],S:[a,function(k){this.milliseconds=100*k}],SS:[f,function(k){this.milliseconds=10*k}],SSS:[/\d{3}/,function(k){this.milliseconds=+k}],s:[g,C("seconds")],ss:[g,C("seconds")],m:[g,C("minutes")],mm:[g,C("minutes")],H:[g,C("hours")],h:[g,C("hours")],HH:[g,C("hours")],hh:[g,C("hours")],D:[g,C("day")],DD:[f,C("day")],Do:[w,function(k){var S=E.ordinal;if(this.day=k.match(/\d+/)[0],S)for(var A=1;A<=31;A+=1)S(A).replace(/\[|\]/g,"")===k&&(this.day=A)}],w:[g,C("week")],ww:[f,C("week")],M:[g,C("month")],MM:[f,C("month")],MMM:[w,function(k){var S=T("months"),A=(T("monthsShort")||S.map((function(F){return F.slice(0,3)}))).indexOf(k)+1;if(A<1)throw Error();this.month=A%12||A}],MMMM:[w,function(k){var S=T("months").indexOf(k)+1;if(S<1)throw Error();this.month=S%12||S}],Y:[/[+-]?\d+/,C("year")],YY:[f,function(k){this.year=_(k)}],YYYY:[/\d{4}/,C("year")],Z:P,ZZ:P};function N(k){for(var S=k,A=E&&E.formats,F=(k=S.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(m,p,s){var u=s&&s.toUpperCase();return p||A[s]||r[s]||A[u].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(l,d,v){return d||v.slice(1)}))}))).match(i),B=F.length,z=0;z<B;z+=1){var Y=F[z],b=R[Y],y=b&&b[0],M=b&&b[1];F[z]=M?{regex:y,parser:M}:Y.replace(/^\[|\]$/g,"")}return function(m){for(var p={},s=0,u=0;s<B;s+=1){var l=F[s];if(typeof l=="string")u+=l.length;else{var d=l.regex,v=l.parser,n=m.slice(u),L=d.exec(n)[0];v.call(p,L),m=m.replace(L,"")}}return(function(o){var I=o.afternoon;if(I!==void 0){var c=o.hours;I?c<12&&(o.hours+=12):c===12&&(o.hours=0),delete o.afternoon}})(p),p}}return function(k,S,A){A.p.customParseFormat=!0,k&&k.parseTwoDigitYear&&(_=k.parseTwoDigitYear);var F=S.prototype,B=F.parse;F.parse=function(z){var Y=z.date,b=z.utc,y=z.args;this.$u=b;var M=y[1];if(typeof M=="string"){var m=y[2]===!0,p=y[3]===!0,s=m||p,u=y[2];p&&(u=y[2]),E=this.$locale(),!m&&u&&(E=A.Ls[u]),this.$d=(function(n,L,o,I){try{if(["x","X"].indexOf(L)>-1)return new Date((L==="X"?1e3:1)*n);var c=N(L)(n),x=c.year,O=c.month,D=c.day,j=c.hours,W=c.minutes,H=c.seconds,nt=c.milliseconds,et=c.zone,bt=c.week,kt=new Date,ct=D||(x||O?1:kt.getDate()),V=x||kt.getFullYear(),it=0;x&&!O||(it=O>0?O-1:kt.getMonth());var Z,U=j||0,rt=W||0,Q=H||0,st=nt||0;return et?new Date(Date.UTC(V,it,ct,U,rt,Q,st+60*et.offset*1e3)):o?new Date(Date.UTC(V,it,ct,U,rt,Q,st)):(Z=new Date(V,it,ct,U,rt,Q,st),bt&&(Z=I(Z).week(bt).toDate()),Z)}catch{return new Date("")}})(Y,M,b,A),this.init(),u&&u!==!0&&(this.$L=this.locale(u).$L),s&&Y!=this.format(M)&&(this.$d=new Date("")),E={}}else if(M instanceof Array)for(var l=M.length,d=1;d<=l;d+=1){y[1]=M[d-1];var v=A.apply(this,y);if(v.isValid()){this.$d=v.$d,this.$L=v.$L,this.init();break}d===l&&(this.$d=new Date(""))}else B.call(this,z)}}}))})),qe=Tt(((t,e)=>{(function(r,i){typeof t=="object"&&e!==void 0?e.exports=i():typeof define=="function"&&define.amd?define(i):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_advancedFormat=i()})(t,(function(){return function(r,i){var a=i.prototype,f=a.format;a.format=function(g){var w=this,E=this.$locale();if(!this.isValid())return f.bind(this)(g);var _=this.$utils(),C=(g||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(P){switch(P){case"Q":return Math.ceil((w.$M+1)/3);case"Do":return E.ordinal(w.$D);case"gggg":return w.weekYear();case"GGGG":return w.isoWeekYear();case"wo":return E.ordinal(w.week(),"W");case"w":case"ww":return _.s(w.week(),P==="w"?1:2,"0");case"W":case"WW":return _.s(w.isoWeek(),P==="W"?1:2,"0");case"k":case"kk":return _.s(String(w.$H===0?24:w.$H),P==="k"?1:2,"0");case"X":return Math.floor(w.$d.getTime()/1e3);case"x":return w.$d.getTime();case"z":return"["+w.offsetName()+"]";case"zzz":return"["+w.offsetName("long")+"]";default:return P}}));return f.bind(this)(C)}}}))})),Xe=Tt(((t,e)=>{(function(r,i){typeof t=="object"&&e!==void 0?e.exports=i():typeof define=="function"&&define.amd?define(i):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_duration=i()})(t,(function(){var r,i,a=1e3,f=6e4,g=36e5,w=864e5,E=31536e6,_=2628e6,C=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,P=/\[([^\]]+)]|YYYY|YY|Y|M{1,2}|D{1,2}|H{1,2}|m{1,2}|s{1,2}|SSS/g,T={years:E,months:_,days:w,hours:g,minutes:f,seconds:a,milliseconds:1,weeks:6048e5},$=function(Y){return Y instanceof B},R=function(Y,b,y){return new B(Y,y,b.$l)},N=function(Y){return i.p(Y)+"s"},k=function(Y){return Y<0},S=function(Y){return k(Y)?Math.ceil(Y):Math.floor(Y)},A=function(Y){return Math.abs(Y)},F=function(Y,b){return Y?k(Y)?{negative:!0,format:""+A(Y)+b}:{negative:!1,format:""+Y+b}:{negative:!1,format:""}},B=(function(){function Y(y,M,m){var p=this;if(this.$d={},this.$l=m,y===void 0&&(this.$ms=0,this.parseFromMilliseconds()),M)return R(y*T[N(M)],this);if(typeof y=="number")return this.$ms=y,this.parseFromMilliseconds(),this;if(typeof y=="object")return Object.keys(y).forEach((function(l){p.$d[N(l)]=y[l]})),this.calMilliseconds(),this;if(typeof y=="string"){var s=y.match(C);if(s){var u=s.slice(2).map((function(l){return l==null?0:Number(l)}));return this.$d.years=u[0],this.$d.months=u[1],this.$d.weeks=u[2],this.$d.days=u[3],this.$d.hours=u[4],this.$d.minutes=u[5],this.$d.seconds=u[6],this.calMilliseconds(),this}}return this}var b=Y.prototype;return b.calMilliseconds=function(){var y=this;this.$ms=Object.keys(this.$d).reduce((function(M,m){return M+(y.$d[m]||0)*T[m]}),0)},b.parseFromMilliseconds=function(){var y=this.$ms;this.$d.years=S(y/E),y%=E,this.$d.months=S(y/_),y%=_,this.$d.days=S(y/w),y%=w,this.$d.hours=S(y/g),y%=g,this.$d.minutes=S(y/f),y%=f,this.$d.seconds=S(y/a),y%=a,this.$d.milliseconds=y},b.toISOString=function(){var y=F(this.$d.years,"Y"),M=F(this.$d.months,"M"),m=+this.$d.days||0;this.$d.weeks&&(m+=7*this.$d.weeks);var p=F(m,"D"),s=F(this.$d.hours,"H"),u=F(this.$d.minutes,"M"),l=this.$d.seconds||0;this.$d.milliseconds&&(l+=this.$d.milliseconds/1e3,l=Math.round(1e3*l)/1e3);var d=F(l,"S"),v=y.negative||M.negative||p.negative||s.negative||u.negative||d.negative,n=s.format||u.format||d.format?"T":"",L=(v?"-":"")+"P"+y.format+M.format+p.format+n+s.format+u.format+d.format;return L==="P"||L==="-P"?"P0D":L},b.toJSON=function(){return this.toISOString()},b.format=function(y){var M=y||"YYYY-MM-DDTHH:mm:ss",m={Y:this.$d.years,YY:i.s(this.$d.years,2,"0"),YYYY:i.s(this.$d.years,4,"0"),M:this.$d.months,MM:i.s(this.$d.months,2,"0"),D:this.$d.days,DD:i.s(this.$d.days,2,"0"),H:this.$d.hours,HH:i.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:i.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:i.s(this.$d.seconds,2,"0"),SSS:i.s(this.$d.milliseconds,3,"0")};return M.replace(P,(function(p,s){return s||String(m[p])}))},b.as=function(y){return this.$ms/T[N(y)]},b.get=function(y){var M=this.$ms,m=N(y);return m==="milliseconds"?M%=1e3:M=m==="weeks"?S(M/T[m]):this.$d[m],M||0},b.add=function(y,M,m){var p;return p=M?y*T[N(M)]:$(y)?y.$ms:R(y,this).$ms,R(this.$ms+p*(m?-1:1),this)},b.subtract=function(y,M){return this.add(y,M,!0)},b.locale=function(y){var M=this.clone();return M.$l=y,M},b.clone=function(){return R(this.$ms,this)},b.humanize=function(y){return r().add(this.$ms,"ms").locale(this.$l).fromNow(!y)},b.valueOf=function(){return this.asMilliseconds()},b.milliseconds=function(){return this.get("milliseconds")},b.asMilliseconds=function(){return this.as("milliseconds")},b.seconds=function(){return this.get("seconds")},b.asSeconds=function(){return this.as("seconds")},b.minutes=function(){return this.get("minutes")},b.asMinutes=function(){return this.as("minutes")},b.hours=function(){return this.get("hours")},b.asHours=function(){return this.as("hours")},b.days=function(){return this.get("days")},b.asDays=function(){return this.as("days")},b.weeks=function(){return this.get("weeks")},b.asWeeks=function(){return this.as("weeks")},b.months=function(){return this.get("months")},b.asMonths=function(){return this.as("months")},b.years=function(){return this.get("years")},b.asYears=function(){return this.as("years")},Y})(),z=function(Y,b,y){return Y.add(b.years()*y,"y").add(b.months()*y,"M").add(b.days()*y,"d").add(b.hours()*y,"h").add(b.minutes()*y,"m").add(b.seconds()*y,"s").add(b.milliseconds()*y,"ms")};return function(Y,b,y){r=y,i=y().$utils(),y.duration=function(p,s){return R(p,{$l:y.locale()},s)},y.isDuration=$;var M=b.prototype.add,m=b.prototype.subtract;b.prototype.add=function(p,s){return $(p)?z(this,p,1):M.bind(this)(p,s)},b.prototype.subtract=function(p,s){return $(p)?z(this,p,-1):m.bind(this)(p,s)}}}))})),Qe=Pe(),X=lt(ee(),1),Je=lt(Ue(),1),Ke=lt(Ze(),1),ti=lt(qe(),1),pt=lt(ee(),1),ei=lt(Xe(),1),Et=(function(){var t=h(function(s,u,l,d){for(l||(l={}),d=s.length;d--;l[s[d]]=u);return l},"o"),e=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],i=[1,27],a=[1,28],f=[1,29],g=[1,30],w=[1,31],E=[1,32],_=[1,33],C=[1,34],P=[1,9],T=[1,10],$=[1,11],R=[1,12],N=[1,13],k=[1,14],S=[1,15],A=[1,16],F=[1,19],B=[1,20],z=[1,21],Y=[1,22],b=[1,23],y=[1,25],M=[1,35],m={trace:h(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:h(function(s,u,l,d,v,n,L){var o=n.length-1;switch(v){case 1:return n[o-1];case 2:this.$=[];break;case 3:n[o-1].push(n[o]),this.$=n[o-1];break;case 4:case 5:this.$=n[o];break;case 6:case 7:this.$=[];break;case 8:d.setWeekday("monday");break;case 9:d.setWeekday("tuesday");break;case 10:d.setWeekday("wednesday");break;case 11:d.setWeekday("thursday");break;case 12:d.setWeekday("friday");break;case 13:d.setWeekday("saturday");break;case 14:d.setWeekday("sunday");break;case 15:d.setWeekend("friday");break;case 16:d.setWeekend("saturday");break;case 17:d.setDateFormat(n[o].substr(11)),this.$=n[o].substr(11);break;case 18:d.enableInclusiveEndDates(),this.$=n[o].substr(18);break;case 19:d.TopAxis(),this.$=n[o].substr(8);break;case 20:d.setAxisFormat(n[o].substr(11)),this.$=n[o].substr(11);break;case 21:d.setTickInterval(n[o].substr(13)),this.$=n[o].substr(13);break;case 22:d.setExcludes(n[o].substr(9)),this.$=n[o].substr(9);break;case 23:d.setIncludes(n[o].substr(9)),this.$=n[o].substr(9);break;case 24:d.setTodayMarker(n[o].substr(12)),this.$=n[o].substr(12);break;case 27:d.setDiagramTitle(n[o].substr(6)),this.$=n[o].substr(6);break;case 28:this.$=n[o].trim(),d.setAccTitle(this.$);break;case 29:case 30:this.$=n[o].trim(),d.setAccDescription(this.$);break;case 31:d.addSection(n[o].substr(8)),this.$=n[o].substr(8);break;case 33:d.addTask(n[o-1],n[o]),this.$="task";break;case 34:this.$=n[o-1],d.setClickEvent(n[o-1],n[o],null);break;case 35:this.$=n[o-2],d.setClickEvent(n[o-2],n[o-1],n[o]);break;case 36:this.$=n[o-2],d.setClickEvent(n[o-2],n[o-1],null),d.setLink(n[o-2],n[o]);break;case 37:this.$=n[o-3],d.setClickEvent(n[o-3],n[o-2],n[o-1]),d.setLink(n[o-3],n[o]);break;case 38:this.$=n[o-2],d.setClickEvent(n[o-2],n[o],null),d.setLink(n[o-2],n[o-1]);break;case 39:this.$=n[o-3],d.setClickEvent(n[o-3],n[o-1],n[o]),d.setLink(n[o-3],n[o-2]);break;case 40:this.$=n[o-1],d.setLink(n[o-1],n[o]);break;case 41:case 47:this.$=n[o-1]+" "+n[o];break;case 42:case 43:case 45:this.$=n[o-2]+" "+n[o-1]+" "+n[o];break;case 44:case 46:this.$=n[o-3]+" "+n[o-2]+" "+n[o-1]+" "+n[o];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:i,14:a,15:f,16:g,17:w,18:E,19:18,20:_,21:C,22:P,23:T,24:$,25:R,26:N,27:k,28:S,29:A,30:F,31:B,33:z,35:Y,36:b,37:24,38:y,40:M},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:36,11:17,12:r,13:i,14:a,15:f,16:g,17:w,18:E,19:18,20:_,21:C,22:P,23:T,24:$,25:R,26:N,27:k,28:S,29:A,30:F,31:B,33:z,35:Y,36:b,37:24,38:y,40:M},t(e,[2,5]),t(e,[2,6]),t(e,[2,17]),t(e,[2,18]),t(e,[2,19]),t(e,[2,20]),t(e,[2,21]),t(e,[2,22]),t(e,[2,23]),t(e,[2,24]),t(e,[2,25]),t(e,[2,26]),t(e,[2,27]),{32:[1,37]},{34:[1,38]},t(e,[2,30]),t(e,[2,31]),t(e,[2,32]),{39:[1,39]},t(e,[2,8]),t(e,[2,9]),t(e,[2,10]),t(e,[2,11]),t(e,[2,12]),t(e,[2,13]),t(e,[2,14]),t(e,[2,15]),t(e,[2,16]),{41:[1,40],43:[1,41]},t(e,[2,4]),t(e,[2,28]),t(e,[2,29]),t(e,[2,33]),t(e,[2,34],{42:[1,42],43:[1,43]}),t(e,[2,40],{41:[1,44]}),t(e,[2,35],{43:[1,45]}),t(e,[2,36]),t(e,[2,38],{42:[1,46]}),t(e,[2,37]),t(e,[2,39])],defaultActions:{},parseError:h(function(s,u){if(u.recoverable)this.trace(s);else{var l=Error(s);throw l.hash=u,l}},"parseError"),parse:h(function(s){var u=this,l=[0],d=[],v=[null],n=[],L=this.table,o="",I=0,c=0,x=0,O=2,D=1,j=n.slice.call(arguments,1),W=Object.create(this.lexer),H={yy:{}};for(var nt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,nt)&&(H.yy[nt]=this.yy[nt]);W.setInput(s,H.yy),H.yy.lexer=W,H.yy.parser=this,W.yylloc===void 0&&(W.yylloc={});var et=W.yylloc;n.push(et);var bt=W.options&&W.options.ranges;typeof H.yy.parseError=="function"?this.parseError=H.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function kt(q){l.length-=2*q,v.length-=q,n.length-=q}h(kt,"popStack");function ct(){var q=d.pop()||W.lex()||D;return typeof q!="number"&&(q instanceof Array&&(d=q,q=d.pop()),q=u.symbols_[q]||q),q}h(ct,"lex");for(var V,it,Z,U,rt,Q={},st,K,Vt,vt;;){if(Z=l[l.length-1],this.defaultActions[Z]?U=this.defaultActions[Z]:(V??(V=ct()),U=L[Z]&&L[Z][V]),U===void 0||!U.length||!U[0]){var Ut="";for(st in vt=[],L[Z])this.terminals_[st]&&st>O&&vt.push("'"+this.terminals_[st]+"'");Ut=W.showPosition?"Parse error on line "+(I+1)+`:
`+W.showPosition()+`
Expecting `+vt.join(", ")+", got '"+(this.terminals_[V]||V)+"'":"Parse error on line "+(I+1)+": Unexpected "+(V==D?"end of input":"'"+(this.terminals_[V]||V)+"'"),this.parseError(Ut,{text:W.match,token:this.terminals_[V]||V,line:W.yylineno,loc:et,expected:vt})}if(U[0]instanceof Array&&U.length>1)throw Error("Parse Error: multiple actions possible at state: "+Z+", token: "+V);switch(U[0]){case 1:l.push(V),v.push(W.yytext),n.push(W.yylloc),l.push(U[1]),V=null,it?(V=it,it=null):(c=W.yyleng,o=W.yytext,I=W.yylineno,et=W.yylloc,x>0&&x--);break;case 2:if(K=this.productions_[U[1]][1],Q.$=v[v.length-K],Q._$={first_line:n[n.length-(K||1)].first_line,last_line:n[n.length-1].last_line,first_column:n[n.length-(K||1)].first_column,last_column:n[n.length-1].last_column},bt&&(Q._$.range=[n[n.length-(K||1)].range[0],n[n.length-1].range[1]]),rt=this.performAction.apply(Q,[o,c,I,H.yy,U[1],v,n].concat(j)),rt!==void 0)return rt;K&&(l=l.slice(0,-1*K*2),v=v.slice(0,-1*K),n=n.slice(0,-1*K)),l.push(this.productions_[U[1]][0]),v.push(Q.$),n.push(Q._$),Vt=L[l[l.length-2]][l[l.length-1]],l.push(Vt);break;case 3:return!0}}return!0},"parse")};m.lexer=(function(){return{EOF:1,parseError:h(function(s,u){if(this.yy.parser)this.yy.parser.parseError(s,u);else throw Error(s)},"parseError"),setInput:h(function(s,u){return this.yy=u||this.yy||{},this._input=s,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:h(function(){var s=this._input[0];return this.yytext+=s,this.yyleng++,this.offset++,this.match+=s,this.matched+=s,s.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),s},"input"),unput:h(function(s){var u=s.length,l=s.split(/(?:\r\n?|\n)/g);this._input=s+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-u),this.offset-=u;var d=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),l.length-1&&(this.yylineno-=l.length-1);var v=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:l?(l.length===d.length?this.yylloc.first_column:0)+d[d.length-l.length].length-l[0].length:this.yylloc.first_column-u},this.options.ranges&&(this.yylloc.range=[v[0],v[0]+this.yyleng-u]),this.yyleng=this.yytext.length,this},"unput"),more:h(function(){return this._more=!0,this},"more"),reject:h(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:h(function(s){this.unput(this.match.slice(s))},"less"),pastInput:h(function(){var s=this.matched.substr(0,this.matched.length-this.match.length);return(s.length>20?"...":"")+s.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:h(function(){var s=this.match;return s.length<20&&(s+=this._input.substr(0,20-s.length)),(s.substr(0,20)+(s.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:h(function(){var s=this.pastInput(),u=Array(s.length+1).join("-");return s+this.upcomingInput()+`
`+u+"^"},"showPosition"),test_match:h(function(s,u){var l,d,v;if(this.options.backtrack_lexer&&(v={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(v.yylloc.range=this.yylloc.range.slice(0))),d=s[0].match(/(?:\r\n?|\n).*/g),d&&(this.yylineno+=d.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:d?d[d.length-1].length-d[d.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+s[0].length},this.yytext+=s[0],this.match+=s[0],this.matches=s,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(s[0].length),this.matched+=s[0],l=this.performAction.call(this,this.yy,this,u,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),l)return l;if(this._backtrack){for(var n in v)this[n]=v[n];return!1}return!1},"test_match"),next:h(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var s,u,l,d;this._more||(this.yytext="",this.match="");for(var v=this._currentRules(),n=0;n<v.length;n++)if(l=this._input.match(this.rules[v[n]]),l&&(!u||l[0].length>u[0].length)){if(u=l,d=n,this.options.backtrack_lexer){if(s=this.test_match(l,v[n]),s!==!1)return s;if(this._backtrack){u=!1;continue}else return!1}else if(!this.options.flex)break}return u?(s=this.test_match(u,v[d]),s===!1?!1:s):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:h(function(){return this.next()||this.lex()},"lex"),begin:h(function(s){this.conditionStack.push(s)},"begin"),popState:h(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:h(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:h(function(s){return s=this.conditionStack.length-1-Math.abs(s||0),s>=0?this.conditionStack[s]:"INITIAL"},"topState"),pushState:h(function(s){this.begin(s)},"pushState"),stateStackSize:h(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:h(function(s,u,l,d){switch(l){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}})();function p(){this.yy={}}return h(p,"Parser"),p.prototype=m,m.Parser=p,new p})();Et.parser=Et;var ii=Et;X.default.extend(Je.default),X.default.extend(Ke.default),X.default.extend(ti.default);var ne={friday:5,saturday:6},J="",Yt="",Ot=void 0,At="",ut=[],ht=[],Lt=new Map,It=[],wt=[],ft="",Ft="",re=["active","done","crit","milestone","vert"],Wt=[],mt="",gt=!1,Pt=!1,Ht="sunday",_t="saturday",zt=0,si=h(function(){It=[],wt=[],ft="",Wt=[],Bt=0,jt=void 0,Dt=void 0,G=[],J="",Yt="",Ft="",Ot=void 0,At="",ut=[],ht=[],gt=!1,Pt=!1,zt=0,Lt=new Map,mt="",Oe(),Ht="sunday",_t="saturday"},"clear"),ni=h(function(t){mt=t},"setDiagramId"),ri=h(function(t){Yt=t},"setAxisFormat"),ai=h(function(){return Yt},"getAxisFormat"),oi=h(function(t){Ot=t},"setTickInterval"),ci=h(function(){return Ot},"getTickInterval"),li=h(function(t){At=t},"setTodayMarker"),di=h(function(){return At},"getTodayMarker"),ui=h(function(t){J=t},"setDateFormat"),hi=h(function(){gt=!0},"enableInclusiveEndDates"),fi=h(function(){return gt},"endDatesAreInclusive"),mi=h(function(){Pt=!0},"enableTopAxis"),yi=h(function(){return Pt},"topAxisEnabled"),ki=h(function(t){Ft=t},"setDisplayMode"),pi=h(function(){return Ft},"getDisplayMode"),gi=h(function(){return J},"getDateFormat"),ae=h((t,e)=>{let r=e.toLowerCase().split(/[\s,]+/).filter(i=>i!=="");return[...new Set([...t,...r])]},"mergeTokens"),bi=h(function(t){ut=ae(ut,t)},"setIncludes"),vi=h(function(){return ut},"getIncludes"),Ti=h(function(t){ht=ae(ht,t)},"setExcludes"),xi=h(function(){return ht},"getExcludes"),$i=h(function(){return Lt},"getLinks"),wi=h(function(t){ft=t,It.push(t)},"addSection"),_i=h(function(){return It},"getSections"),Di=h(function(){let t=he(),e=0;for(;!t&&e<10;)t=he(),e++;return wt=G,wt},"getTasks"),oe=h(function(t,e,r,i){let a=t.format(e.trim()),f=t.format("YYYY-MM-DD");return i.includes(a)||i.includes(f)?!1:r.includes("weekends")&&(t.isoWeekday()===ne[_t]||t.isoWeekday()===ne[_t]+1)||r.includes(t.format("dddd").toLowerCase())?!0:r.includes(a)||r.includes(f)},"isInvalidDate"),Si=h(function(t){Ht=t},"setWeekday"),Mi=h(function(){return Ht},"getWeekday"),Ci=h(function(t){_t=t},"setWeekend"),ce=h(function(t,e,r,i){if(!r.length||t.manualEndTime)return;let a;a=t.startTime instanceof Date?(0,X.default)(t.startTime):(0,X.default)(t.startTime,e,!0),a=a.add(1,"d");let f;f=t.endTime instanceof Date?(0,X.default)(t.endTime):(0,X.default)(t.endTime,e,!0);let[g,w]=Ei(a,f,e,r,i);t.endTime=g.toDate(),t.renderEndTime=w},"checkTaskDates"),Ei=h(function(t,e,r,i,a){let f=!1,g=null,w=e.add(1e4,"d");for(;t<=e;){if(f||(g=e.toDate()),f=oe(t,r,i,a),f&&(e=e.add(1,"d"),e>w))throw Error("Failed to find a valid date that was not excluded by `excludes` after 10,000 iterations.");t=t.add(1,"d")}return[e,g]},"fixTaskDates"),Nt=h(function(t,e,r){if(r=r.trim(),h(f=>{let g=f.trim();return g==="x"||g==="X"},"isTimestampFormat")(e)&&/^\d+$/.test(r))return new Date(Number(r));let i=/^after\s+(?<ids>[\d\w- ]+)/.exec(r);if(i!==null){let f=null;for(let w of i.groups.ids.split(" ")){let E=ot(w);E!==void 0&&(!f||E.endTime>f.endTime)&&(f=E)}if(f)return f.endTime;let g=new Date;return g.setHours(0,0,0,0),g}let a=(0,X.default)(r,e.trim(),!0);if(a.isValid())return a.toDate();{at.debug("Invalid date:"+r),at.debug("With date format:"+e.trim());let f=new Date(r);if(f===void 0||isNaN(f.getTime())||f.getFullYear()<-1e4||f.getFullYear()>1e4)throw Error("Invalid date:"+r);return f}},"getStartDate"),le=h(function(t){let e=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return e===null?[NaN,"ms"]:[Number.parseFloat(e[1]),e[2]]},"parseDuration"),de=h(function(t,e,r,i=!1){r=r.trim();let a=/^until\s+(?<ids>[\d\w- ]+)/.exec(r);if(a!==null){let _=null;for(let P of a.groups.ids.split(" ")){let T=ot(P);T!==void 0&&(!_||T.startTime<_.startTime)&&(_=T)}if(_)return _.startTime;let C=new Date;return C.setHours(0,0,0,0),C}let f=(0,X.default)(r,e.trim(),!0);if(f.isValid())return i&&(f=f.add(1,"d")),f.toDate();let g=(0,X.default)(t),[w,E]=le(r);if(!Number.isNaN(w)){let _=g.add(w,E);_.isValid()&&(g=_)}return g.toDate()},"getEndDate"),Bt=0,yt=h(function(t){return t===void 0?(Bt+=1,"task"+Bt):t},"parseId"),Yi=h(function(t,e){let r;r=e.substr(0,1)===":"?e.substr(1,e.length):e;let i=r.split(","),a={};Rt(i,a,re);for(let g=0;g<i.length;g++)i[g]=i[g].trim();let f="";switch(i.length){case 1:a.id=yt(),a.startTime=t.endTime,f=i[0];break;case 2:a.id=yt(),a.startTime=Nt(void 0,J,i[0]),f=i[1];break;case 3:a.id=yt(i[0]),a.startTime=Nt(void 0,J,i[1]),f=i[2];break;default:}return f&&(a.endTime=de(a.startTime,J,f,gt),a.manualEndTime=(0,X.default)(f,"YYYY-MM-DD",!0).isValid(),ce(a,J,ht,ut)),a},"compileData"),Oi=h(function(t,e){let r;r=e.substr(0,1)===":"?e.substr(1,e.length):e;let i=r.split(","),a={};Rt(i,a,re);for(let f=0;f<i.length;f++)i[f]=i[f].trim();switch(i.length){case 1:a.id=yt(),a.startTime={type:"prevTaskEnd",id:t},a.endTime={data:i[0]};break;case 2:a.id=yt(),a.startTime={type:"getStartDate",startData:i[0]},a.endTime={data:i[1]};break;case 3:a.id=yt(i[0]),a.startTime={type:"getStartDate",startData:i[1]},a.endTime={data:i[2]};break;default:}return a},"parseData"),jt,Dt,G=[],ue={},Ai=h(function(t,e){let r={section:ft,type:ft,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:e},task:t,classes:[]},i=Oi(Dt,e);r.raw.startTime=i.startTime,r.raw.endTime=i.endTime,r.id=i.id,r.prevTaskId=Dt,r.active=i.active,r.done=i.done,r.crit=i.crit,r.milestone=i.milestone,r.vert=i.vert,r.vert?r.order=-1:(r.order=zt,zt++);let a=G.push(r);Dt=r.id,ue[r.id]=a-1},"addTask"),ot=h(function(t){let e=ue[t];return G[e]},"findTaskById"),Li=h(function(t,e){let r={section:ft,type:ft,description:t,task:t,classes:[]},i=Yi(jt,e);r.startTime=i.startTime,r.endTime=i.endTime,r.id=i.id,r.active=i.active,r.done=i.done,r.crit=i.crit,r.milestone=i.milestone,r.vert=i.vert,jt=r,wt.push(r)},"addTaskOrg"),he=h(function(){let t=h(function(r){let i=G[r],a="";switch(G[r].raw.startTime.type){case"prevTaskEnd":i.startTime=ot(i.prevTaskId).endTime;break;case"getStartDate":a=Nt(void 0,J,G[r].raw.startTime.startData),a&&(G[r].startTime=a);break}return G[r].startTime&&(G[r].endTime=de(G[r].startTime,J,G[r].raw.endTime.data,gt),G[r].endTime&&(G[r].processed=!0,G[r].manualEndTime=(0,X.default)(G[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),ce(G[r],J,ht,ut))),G[r].processed},"compileTask"),e=!0;for(let[r,i]of G.entries())t(r),e&&(e=i.processed);return e},"compileTasks"),Ii=h(function(t,e){let r=e;dt().securityLevel!=="loose"&&(r=(0,Qe.sanitizeUrl)(e)),t.split(",").forEach(function(i){ot(i)!==void 0&&(me(i,()=>{window.open(r,"_self")}),Lt.set(i,r))}),fe(t,"clickable")},"setLink"),fe=h(function(t,e){t.split(",").forEach(function(r){let i=ot(r);i!==void 0&&i.classes.push(e)})},"setClass"),Fi=h(function(t,e,r){if(dt().securityLevel!=="loose"||e===void 0)return;let i=[];if(typeof r=="string"){i=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let a=0;a<i.length;a++){let f=i[a].trim();f.startsWith('"')&&f.endsWith('"')&&(f=f.substr(1,f.length-2)),i[a]=f}}i.length===0&&i.push(t),ot(t)!==void 0&&me(t,()=>{Se.runFunc(e,...i)})},"setClickFun"),me=h(function(t,e){Wt.push(function(){let r=mt?`${mt}-${t}`:t,i=document.querySelector(`[id="${r}"]`);i!==null&&i.addEventListener("click",function(){e()})},function(){let r=mt?`${mt}-${t}`:t,i=document.querySelector(`[id="${r}-text"]`);i!==null&&i.addEventListener("click",function(){e()})})},"pushFun"),Wi={getConfig:h(()=>dt().gantt,"getConfig"),clear:si,setDateFormat:ui,getDateFormat:gi,enableInclusiveEndDates:hi,endDatesAreInclusive:fi,enableTopAxis:mi,topAxisEnabled:yi,setAxisFormat:ri,getAxisFormat:ai,setTickInterval:oi,getTickInterval:ci,setTodayMarker:li,getTodayMarker:di,setAccTitle:Ye,getAccTitle:We,setDiagramTitle:Ee,getDiagramTitle:Fe,setDiagramId:ni,setDisplayMode:ki,getDisplayMode:pi,setAccDescription:Ce,getAccDescription:Ie,addSection:wi,getSections:_i,getTasks:Di,addTask:Ai,findTaskById:ot,addTaskOrg:Li,setIncludes:bi,getIncludes:vi,setExcludes:Ti,getExcludes:xi,setClickEvent:h(function(t,e,r){t.split(",").forEach(function(i){Fi(i,e,r)}),fe(t,"clickable")},"setClickEvent"),setLink:Ii,getLinks:$i,bindFunctions:h(function(t){Wt.forEach(function(e){e(t)})},"bindFunctions"),parseDuration:le,isInvalidDate:oe,setWeekday:Si,getWeekday:Mi,setWeekend:Ci};function Rt(t,e,r){let i=!0;for(;i;)i=!1,r.forEach(function(a){let f="^\\s*"+a+"\\s*$",g=new RegExp(f);t[0].match(g)&&(e[a]=!0,t.shift(1),i=!0)})}h(Rt,"getTaskTags"),pt.default.extend(ei.default);var Pi=h(function(){at.debug("Something is calling, setConf, remove the call")},"setConf"),ye={monday:ve,tuesday:$e,wednesday:xe,thursday:we,friday:De,saturday:Te,sunday:_e},Hi=h((t,e)=>{let r=[...t].map(()=>-1/0),i=[...t].sort((f,g)=>f.startTime-g.startTime||f.order-g.order),a=0;for(let f of i)for(let g=0;g<r.length;g++)if(f.startTime>=r[g]){r[g]=f.endTime,f.order=g+e,g>a&&(a=g);break}return a},"getMaxIntersections"),tt,Gt=1e4,zi={parser:ii,db:Wi,renderer:{setConf:Pi,draw:h(function(t,e,r,i){let a=dt().gantt;i.db.setDiagramId(e);let f=dt().securityLevel,g;f==="sandbox"&&(g=St("#i"+e));let w=St(f==="sandbox"?g.nodes()[0].contentDocument.body:"body"),E=f==="sandbox"?g.nodes()[0].contentDocument:document,_=E.getElementById(e);tt=_.parentElement.offsetWidth,tt===void 0&&(tt=1200),a.useWidth!==void 0&&(tt=a.useWidth);let C=i.db.getTasks(),P=C.filter(m=>!m.vert),T=[];for(let m of P)T.push(m.type);T=M(T);let $={},R=2*a.topPadding;if(i.db.getDisplayMode()==="compact"||a.displayMode==="compact"){let m={};for(let s of P)m[s.section]===void 0?m[s.section]=[s]:m[s.section].push(s);let p=0;for(let s of Object.keys(m)){let u=Hi(m[s],p)+1;p+=u,R+=u*(a.barHeight+a.barGap),$[s]=u}}else{R+=P.length*(a.barHeight+a.barGap);for(let m of T)$[m]=P.filter(p=>p.type===m).length}_.setAttribute("viewBox","0 0 "+tt+" "+R);let N=w.select(`[id="${e}"]`),k=pe().domain([ge(C,function(m){return m.startTime}),be(C,function(m){return m.endTime})]).rangeRound([0,tt-a.leftPadding-a.rightPadding]);function S(m,p){let s=m.startTime,u=p.startTime,l=0;return s>u?l=1:s<u&&(l=-1),l}h(S,"taskCompare"),C.sort(S),A(C,tt,R),Ae(N,R,tt,a.useMaxWidth),N.append("text").text(i.db.getDiagramTitle()).attr("x",tt/2).attr("y",a.titleTopMargin).attr("class","titleText");function A(m,p,s){let u=a.barHeight,l=u+a.barGap,d=a.topPadding,v=a.leftPadding,n=ke().domain([0,T.length]).range(["#00B9FA","#F95002"]).interpolate(Me);B(l,d,v,p,s,m,i.db.getExcludes(),i.db.getIncludes()),Y(v,d,p,s),F(m,l,d,v,u,n,p,s),b(l,d,v,u,n),y(v,d,p,s)}h(A,"makeGantt");function F(m,p,s,u,l,d,v){m.sort((c,x)=>c.vert===x.vert?0:c.vert?1:-1);let n=m.filter(c=>!c.vert),L=[...new Set(n.map(c=>c.order))].map(c=>n.find(x=>x.order===c));N.append("g").selectAll("rect").data(L).enter().append("rect").attr("x",0).attr("y",function(c,x){return x=c.order,x*p+s-2}).attr("width",function(){return v-a.rightPadding/2}).attr("height",p).attr("class",function(c){for(let[x,O]of T.entries())if(c.type===O)return"section section"+x%a.numberSectionStyles;return"section section0"}).enter();let o=N.append("g").selectAll("rect").data(m).enter(),I=i.db.getLinks();if(o.append("rect").attr("id",function(c){return e+"-"+c.id}).attr("rx",3).attr("ry",3).attr("x",function(c){return c.milestone?k(c.startTime)+u+.5*(k(c.endTime)-k(c.startTime))-.5*l:k(c.startTime)+u}).attr("y",function(c,x){return x=c.order,c.vert?a.gridLineStartPadding:x*p+s}).attr("width",function(c){return c.milestone?l:c.vert?.08*l:k(c.renderEndTime||c.endTime)-k(c.startTime)}).attr("height",function(c){return c.vert?n.length*(a.barHeight+a.barGap)+a.barHeight*2:l}).attr("transform-origin",function(c,x){return x=c.order,(k(c.startTime)+u+.5*(k(c.endTime)-k(c.startTime))).toString()+"px "+(x*p+s+.5*l).toString()+"px"}).attr("class",function(c){let x="";c.classes.length>0&&(x=c.classes.join(" "));let O=0;for(let[j,W]of T.entries())c.type===W&&(O=j%a.numberSectionStyles);let D="";return c.active?c.crit?D+=" activeCrit":D=" active":c.done?D=c.crit?" doneCrit":" done":c.crit&&(D+=" crit"),D.length===0&&(D=" task"),c.milestone&&(D=" milestone "+D),c.vert&&(D=" vert "+D),D+=O,D+=" "+x,"task"+D}),o.append("text").attr("id",function(c){return e+"-"+c.id+"-text"}).text(function(c){return c.task}).attr("font-size",a.fontSize).attr("x",function(c){let x=k(c.startTime),O=k(c.renderEndTime||c.endTime);if(c.milestone&&(x+=.5*(k(c.endTime)-k(c.startTime))-.5*l,O=x+l),c.vert)return k(c.startTime)+u;let D=this.getBBox().width;return D>O-x?O+D+1.5*a.leftPadding>v?x+u-5:O+u+5:(O-x)/2+x+u}).attr("y",function(c,x){return c.vert?a.gridLineStartPadding+n.length*(a.barHeight+a.barGap)+60:(x=c.order,x*p+a.barHeight/2+(a.fontSize/2-2)+s)}).attr("text-height",l).attr("class",function(c){let x=k(c.startTime),O=k(c.endTime);c.milestone&&(O=x+l);let D=this.getBBox().width,j="";c.classes.length>0&&(j=c.classes.join(" "));let W=0;for(let[nt,et]of T.entries())c.type===et&&(W=nt%a.numberSectionStyles);let H="";return c.active&&(H=c.crit?"activeCritText"+W:"activeText"+W),c.done?H=c.crit?H+" doneCritText"+W:H+" doneText"+W:c.crit&&(H=H+" critText"+W),c.milestone&&(H+=" milestoneText"),c.vert&&(H+=" vertText"),D>O-x?O+D+1.5*a.leftPadding>v?j+" taskTextOutsideLeft taskTextOutside"+W+" "+H:j+" taskTextOutsideRight taskTextOutside"+W+" "+H+" width-"+D:j+" taskText taskText"+W+" "+H+" width-"+D}),dt().securityLevel==="sandbox"){let c;c=St("#i"+e);let x=c.nodes()[0].contentDocument;o.filter(function(O){return I.has(O.id)}).each(function(O){var D=x.querySelector("#"+CSS.escape(e+"-"+O.id)),j=x.querySelector("#"+CSS.escape(e+"-"+O.id+"-text"));let W=D.parentNode;var H=x.createElement("a");H.setAttribute("xlink:href",I.get(O.id)),H.setAttribute("target","_top"),W.appendChild(H),H.appendChild(D),H.appendChild(j)})}}h(F,"drawRects");function B(m,p,s,u,l,d,v,n){if(v.length===0&&n.length===0)return;let L,o;for(let{startTime:D,endTime:j}of d)(L===void 0||D<L)&&(L=D),(o===void 0||j>o)&&(o=j);if(!L||!o)return;if((0,pt.default)(o).diff((0,pt.default)(L),"year")>5){at.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let I=i.db.getDateFormat(),c=[],x=null,O=(0,pt.default)(L);for(;O.valueOf()<=o;)i.db.isInvalidDate(O,I,v,n)?x?x.end=O:x={start:O,end:O}:x&&(x=(c.push(x),null)),O=O.add(1,"d");N.append("g").selectAll("rect").data(c).enter().append("rect").attr("id",D=>e+"-exclude-"+D.start.format("YYYY-MM-DD")).attr("x",D=>k(D.start.startOf("day"))+s).attr("y",a.gridLineStartPadding).attr("width",D=>k(D.end.endOf("day"))-k(D.start.startOf("day"))).attr("height",l-p-a.gridLineStartPadding).attr("transform-origin",function(D,j){return(k(D.start)+s+.5*(k(D.end)-k(D.start))).toString()+"px "+(j*m+.5*l).toString()+"px"}).attr("class","exclude-range")}h(B,"drawExcludeDays");function z(m,p,s,u){if(s<=0||m>p)return 1/0;let l=p-m,d=pt.default.duration({[u??"day"]:s}).asMilliseconds();return d<=0?1/0:Math.ceil(l/d)}h(z,"getEstimatedTickCount");function Y(m,p,s,u){let l=i.db.getDateFormat(),d=i.db.getAxisFormat(),v;v=d||(l==="D"?"%d":a.axisFormat??"%Y-%m-%d");let n=Ve(k).tickSize(-u+p+a.gridLineStartPadding).tickFormat(Jt(v)),L=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(i.db.getTickInterval()||a.tickInterval);if(L!==null){let o=parseInt(L[1],10);if(isNaN(o)||o<=0)at.warn(`Invalid tick interval value: "${L[1]}". Skipping custom tick interval.`);else{let I=L[2],c=i.db.getWeekday()||a.weekday,x=k.domain(),O=x[0],D=x[1],j=z(O,D,o,I);if(j>Gt)at.warn(`The tick interval "${o}${I}" would generate ${j} ticks, which exceeds the maximum allowed (${Gt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(I){case"millisecond":n.ticks(qt.every(o));break;case"second":n.ticks(Xt.every(o));break;case"minute":n.ticks(Zt.every(o));break;case"hour":n.ticks(te.every(o));break;case"day":n.ticks(Kt.every(o));break;case"week":n.ticks(ye[c].every(o));break;case"month":n.ticks(Qt.every(o));break}}}if(N.append("g").attr("class","grid").attr("transform","translate("+m+", "+(u-50)+")").call(n).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),i.db.topAxisEnabled()||a.topAxis){let o=Ge(k).tickSize(-u+p+a.gridLineStartPadding).tickFormat(Jt(v));if(L!==null){let I=parseInt(L[1],10);if(isNaN(I)||I<=0)at.warn(`Invalid tick interval value: "${L[1]}". Skipping custom tick interval.`);else{let c=L[2],x=i.db.getWeekday()||a.weekday,O=k.domain(),D=O[0],j=O[1];if(z(D,j,I,c)<=Gt)switch(c){case"millisecond":o.ticks(qt.every(I));break;case"second":o.ticks(Xt.every(I));break;case"minute":o.ticks(Zt.every(I));break;case"hour":o.ticks(te.every(I));break;case"day":o.ticks(Kt.every(I));break;case"week":o.ticks(ye[x].every(I));break;case"month":o.ticks(Qt.every(I));break}}}N.append("g").attr("class","grid").attr("transform","translate("+m+", "+p+")").call(o).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}h(Y,"makeGrid");function b(m,p){let s=0,u=Object.keys($).map(l=>[l,$[l]]);N.append("g").selectAll("text").data(u).enter().append(function(l){let d=l[0].split(Le.lineBreakRegex),v=-(d.length-1)/2,n=E.createElementNS("http://www.w3.org/2000/svg","text");n.setAttribute("dy",v+"em");for(let[L,o]of d.entries()){let I=E.createElementNS("http://www.w3.org/2000/svg","tspan");I.setAttribute("alignment-baseline","central"),I.setAttribute("x","10"),L>0&&I.setAttribute("dy","1em"),I.textContent=o,n.appendChild(I)}return n}).attr("x",10).attr("y",function(l,d){if(d>0)for(let v=0;v<d;v++)return s+=u[d-1][1],l[1]*m/2+s*m+p;else return l[1]*m/2+p}).attr("font-size",a.sectionFontSize).attr("class",function(l){for(let[d,v]of T.entries())if(l[0]===v)return"sectionTitle sectionTitle"+d%a.numberSectionStyles;return"sectionTitle"})}h(b,"vertLabels");function y(m,p,s,u){let l=i.db.getTodayMarker();if(l==="off")return;let d=N.append("g").attr("class","today"),v=new Date,n=d.append("line");n.attr("x1",k(v)+m).attr("x2",k(v)+m).attr("y1",a.titleTopMargin).attr("y2",u-a.titleTopMargin).attr("class","today"),l!==""&&n.attr("style",l.replace(/,/g,";"))}h(y,"drawToday");function M(m){let p={},s=[];for(let u=0,l=m.length;u<l;++u)Object.prototype.hasOwnProperty.call(p,m[u])||(p[m[u]]=!0,s.push(m[u]));return s}h(M,"checkUnique")},"draw")},styles:h(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done task text displayed outside the bar sits against the diagram background,
     not against the done-task bar, so it must use the outside/contrast color. */
  .doneText0.taskTextOutsideLeft,
  .doneText0.taskTextOutsideRight,
  .doneText1.taskTextOutsideLeft,
  .doneText1.taskTextOutsideRight,
  .doneText2.taskTextOutsideLeft,
  .doneText2.taskTextOutsideRight,
  .doneText3.taskTextOutsideLeft,
  .doneText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done-crit task text outside the bar \u2014 same reasoning as doneText above. */
  .doneCritText0.taskTextOutsideLeft,
  .doneCritText0.taskTextOutsideRight,
  .doneCritText1.taskTextOutsideLeft,
  .doneCritText1.taskTextOutsideRight,
  .doneCritText2.taskTextOutsideLeft,
  .doneCritText2.taskTextOutsideRight,
  .doneCritText3.taskTextOutsideLeft,
  .doneCritText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles")};export{zi as diagram};
