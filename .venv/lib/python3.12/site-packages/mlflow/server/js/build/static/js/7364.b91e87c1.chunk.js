"use strict";(self.webpackChunk_mlflow_mlflow=self.webpackChunk_mlflow_mlflow||[]).push([[7364],{87364:function(e,n,t){t.d(n,{r:function(){return Q}});var a=t(89555),i=t(9133),r=t(68048),o=t(16934),s=t(31014),l=t(48624);const d="compareToRunUuid",u=()=>{var e;const[n,t]=(0,l.ok)();return[null!==(e=n.get(d))&&void 0!==e?e:void 0,(0,s.useCallback)((e=>{t((n=>void 0===e?(n.delete(d),n):(n.set(d,e),n)))}),[t])]};var c=t(76010),m=t(39330),p=t(70403),g=t(7450),v=t(58481),f=t(24759),h=t(79432),y=t(50111);var x={name:"a41n9l",styles:"justify-content:flex-start !important"},T={name:"0",styles:""},I={name:"bcffy2",styles:"display:flex;align-items:center;justify-content:space-between"},b={name:"fhxb3m",styles:"display:flex;flex-direction:row;align-items:center"},C={name:"a41n9l",styles:"justify-content:flex-start !important"};const Y=({experimentId:e,currentRunUuid:n,setCompareToRunUuid:t,compareToRunUuid:i,setCurrentRunUuid:l})=>{const{theme:d}=(0,o.u)(),u=(0,g.tz)(),c=(0,p.LE)(),{runInfos:Y}=(0,f.Xz)(e),U=(0,s.useMemo)((()=>{if(Y)return Y.map((e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}})).filter((e=>e.key))}),[Y]),S=(0,s.useMemo)((()=>{if(Y)return Y.filter((e=>e.runUuid!==n)).map((e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}})).filter((e=>Boolean(e.key)))}),[Y,n]),w=null===Y||void 0===Y?void 0:Y.find((e=>e.runUuid===n)),R=null===Y||void 0===Y?void 0:Y.find((e=>e.runUuid===i)),N=(0,s.useCallback)((n=>{const t=v.Ay.getRunPageRoute(e,n)+"/evaluations",a=new URLSearchParams(window.location.search),i=new URL(t,window.location.origin);a.forEach(((e,n)=>{i.searchParams.set(n,e)})),window.location.href=i.toString()}),[e]),A=null!==l&&void 0!==l?l:N;return n?(0,y.FD)("div",{css:(0,a.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)("div",{css:(0,a.AH)({display:"flex",alignItems:"center",justifyContent:"flex-start",gap:d.spacing.sm},""),children:(0,y.FD)(r.DialogCombobox,{componentId:f.WB,id:"compare-to-run-combobox",value:n?[n]:void 0,children:[(0,y.Y)(r.DialogComboboxCustomButtonTriggerWrapper,{children:(0,y.Y)(o.B,{endIcon:(0,y.Y)(r.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:x,children:(0,y.FD)("div",{css:(0,a.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center",fontSize:`${d.typography.fontSizeSm}px !important`},""),children:[(0,y.Y)(h.E,{color:c(n)}),null!==w&&void 0!==w&&w.runName?(0,y.Y)(o.T.Hint,{children:null===w||void 0===w?void 0:w.runName}):u.formatMessage({id:"PUQxu5",defaultMessage:"Select baseline run"})]})})}),(0,y.Y)(r.DialogComboboxContent,{children:(0,y.Y)(r.DialogComboboxOptionList,{children:(U||[]).map(((e,t)=>(0,y.Y)(r.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>A(e.key),checked:e.key===n,children:(0,y.FD)("div",{css:(0,a.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)(h.E,{color:c(e.key)}),e.value]})},t)))})})]})}),(0,y.Y)("span",{css:T,children:u.formatMessage({id:"iYmFCZ",defaultMessage:"compare to"})}),t&&(0,y.Y)("div",{css:I,children:(0,y.FD)("div",{css:b,children:[(0,y.FD)(r.DialogCombobox,{componentId:f.WB,id:"compare-to-run-combobox",value:i?[i]:void 0,children:[(0,y.Y)(r.DialogComboboxCustomButtonTriggerWrapper,{children:(0,y.Y)(o.B,{endIcon:(0,y.Y)(r.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:C,children:(0,y.Y)("div",{css:(0,a.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center",fontSize:`${d.typography.fontSizeSm}px !important`},""),children:null!==R&&void 0!==R&&R.runName?(0,y.FD)(y.FK,{children:[(0,y.Y)(h.E,{color:c(i)}),(0,y.Y)(o.T.Hint,{children:null===R||void 0===R?void 0:R.runName})]}):(0,y.Y)("span",{css:(0,a.AH)({color:d.colors.textPlaceholder},""),children:u.formatMessage({id:"XkpMf+",defaultMessage:"baseline run"})})})})}),(0,y.Y)(r.DialogComboboxContent,{children:(0,y.Y)(r.DialogComboboxOptionList,{children:(S||[]).map(((e,n)=>(0,y.Y)(r.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>t(e.key),checked:e.key===i,children:(0,y.FD)("div",{css:(0,a.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)(h.E,{color:c(e.key)}),e.value]})},n)))})})]}),(null===R||void 0===R?void 0:R.runName)&&(0,y.Y)(m.X,{"aria-hidden":"false",css:(0,a.AH)({color:d.colors.textPlaceholder,fontSize:d.typography.fontSizeSm,marginLeft:d.spacing.sm,":hover":{color:d.colors.actionTertiaryTextHover}},""),role:"button",onClick:()=>{t(void 0)},onPointerDownCapture:e=>{e.stopPropagation()}})]})})]}):(0,y.Y)(y.FK,{})};var U=t(91144),S=t(86443),w=t(38566);const R=e=>(0,s.useMemo)((()=>e?(0,i.intersection)((0,w.T)(e),[f.o8.Evaluations,f.o8.Metrics,f.o8.Assessments]):[]),[e]);var N=t(11473),A=t(2250),D=t(10939),k=t(88443),E=t(10811),_=t(26809),M=t(17111),F=t(77484),L=t(7204);const $="_assessments.json",H=()=>{const e=(0,E.wA)(),[n,t]=(0,s.useState)(!1);return{savePendingAssessments:(0,s.useCallback)((async(n,a,r)=>{try{t(!0);const o=await(async e=>{const n=(0,F.To)($,e),t=await(0,F.Up)(n).then((e=>JSON.parse(e)));if(!(0,i.isArray)(t.data)||!(0,i.isArray)(t.columns))throw new Error("Artifact is malformed and/or not valid JSON");return t})(n),s=((e,n)=>n.map((n=>{var t,a,i;return[e,n.name,{source_type:null===(t=n.source)||void 0===t?void 0:t.sourceType,source_id:null===(a=n.source)||void 0===a?void 0:a.sourceId,source_metadata:null===(i=n.source)||void 0===i?void 0:i.metadata},n.timestamp||null,n.booleanValue||null,n.numericValue||null,n.stringValue||null,n.rationale||null,n.metadata||null,null,null]})))(a,r),l=((e,n,t)=>{const a=(0,M.G4)($,n),r=t.map((({name:e,source:n})=>({name:e,source:n?{source_type:n.sourceType,source_id:n.sourceId,source_metadata:n.metadata}:void 0}))),o=a.entries.filter((({evaluation_id:n,name:t,source:a})=>e===n&&r.find((e=>(0,i.isEqual)({name:t,source:a},e))))).map((e=>a.entries.indexOf(e)));return n.data.filter(((e,n)=>!o.includes(n)))})(a,o,r);await e((0,_.Of)(n,$,{columns:o.columns,data:[...l,...s]})),e({type:(0,L.ec)(_.So),payload:(0,M.G4)($,{columns:o.columns,data:[...l,...s]}),meta:{runUuid:n,artifactPath:$}})}catch(o){throw c.A.logErrorAndNotifyUser(o.message||o),o}finally{t(!1)}}),[e]),isSaving:n}};var P=t(74823);const O=P.J1`
  query SearchRuns($data: MlflowSearchRunsInput!) {
    mlflowSearchRuns(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      runs {
        info {
          runName
          status
          runUuid
          experimentId
          artifactUri
          endTime
          lifecycleStage
          startTime
          userId
        }
        experiment {
          experimentId
          name
          tags {
            key
            value
          }
          artifactLocation
          lifecycleStage
          lastUpdateTime
        }
        data {
          metrics {
            key
            value
            step
            timestamp
          }
          params {
            key
            value
          }
          tags {
            key
            value
          }
        }
        inputs {
          datasetInputs {
            dataset {
              digest
              name
              profile
              schema
              source
              sourceType
            }
            tags {
              key
              value
            }
          }
          modelInputs {
            modelId
          }
        }
        outputs {
          modelOutputs {
            modelId
            step
          }
        }
        modelVersions {
          version
          name
          creationTimestamp
          status
          source
        }
      }
    }
  }
`,j=({filter:e,experimentIds:n,disabled:t=!1})=>{var a,r,o;const{data:s,loading:l,error:d,refetch:u}=(0,P.IT)(O,{variables:{data:{filter:e,experimentIds:n}},skip:t});return{loading:l,data:(0,i.first)(null!==(a=null===s||void 0===s||null===(r=s.mlflowSearchRuns)||void 0===r?void 0:r.runs)&&void 0!==a?a:[]),refetchRun:u,apolloError:d,apiError:null===s||void 0===s||null===(o=s.mlflowSearchRuns)||void 0===o?void 0:o.apiError}};var B={name:"r3950p",styles:"flex:1;display:flex;justify-content:center;align-items:center"};const z=({experimentId:e,runUuid:n,runTags:t,runDisplayName:i,data:s})=>{const{theme:l}=(0,o.u)(),d=R(t),c=0===(null===s||void 0===s?void 0:s.length),[m,p]=u(),g=(0,N.N9)(),v=H(),{data:h,displayName:x,loading:T}=V(e,m,d);if(T)return(0,y.Y)(r.LegacySkeleton,{});const I=e=>e.filter((e=>e.type===f.$6.ASSESSMENT||e.type===f.$6.INPUT||e.type===f.$6.TRACE_INFO&&[f.tj,f.$W,f.Pn].includes(e.id)));return c?(0,y.Y)("div",{css:B,children:(0,y.Y)(r.Empty,{title:(0,y.Y)(k.A,{id:"NqqMPs",defaultMessage:"No evaluation tables logged"}),description:null})}):(0,y.FD)("div",{css:(0,a.AH)({marginTop:l.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[(0,y.Y)("div",{css:(0,a.AH)({width:"100%",padding:`${l.spacing.xs}px 0`},""),children:(0,y.Y)(Y,{experimentId:e,currentRunUuid:n,compareToRunUuid:m,setCompareToRunUuid:p})}),(()=>{const t={experimentId:e,currentRunDisplayName:i,currentEvaluationResults:s||[],compareToEvaluationResults:h,runUuid:n,compareToRunUuid:m,compareToRunDisplayName:x,compareToRunLoading:T,saveAssessmentsQuery:v,getTrace:S.R,initialSelectedColumns:I};return(0,y.Y)(f.tU,{makeHtml:g,children:(0,y.Y)(f.js,{...t})})})()]})},V=(e,n,t)=>{const{data:a,isLoading:r}=(0,f.Ie)({runUuid:n||"",artifacts:t},{disabled:(0,i.isNil)(n)}),{data:o,loading:s}=j({experimentIds:[e],filter:`attributes.runId = "${n}"`,disabled:(0,i.isNil)(n)});return{data:a,displayName:c.A.getRunDisplayName(null===o||void 0===o?void 0:o.info,n),loading:r||s}};var W=t(27288);var q={name:"1nxh63r",styles:"overflow-y:hidden;height:100%;display:flex;flex-direction:column"};const K=({experimentId:e,runUuid:n,runTags:t,runDisplayName:i,setCurrentRunUuid:r})=>{const{theme:l}=(0,o.u)(),d=(0,N.N9)(),[c,m]=u(),{assessmentInfos:g,allColumns:v,totalCount:h,isLoading:x,error:T,tableFilterOptions:I}=(0,f.KW)({experimentId:e,runUuid:n,otherRunUuid:c}),[b,C]=(0,s.useState)(""),[w,R]=(0,f.R7)(),k=(0,p.LE)(),E=(0,W.jE)(),_=(0,s.useCallback)((e=>e.filter((e=>e.type===f.$6.ASSESSMENT||e.type===f.$6.EXPECTATION||e.type===f.$6.INPUT||e.type===f.$6.TRACE_INFO&&[f.tj,f.Rl,f.$W,f.YO].includes(e.id)))),[]),{selectedColumns:M,toggleColumns:F,setSelectedColumns:L}=(0,f.K0)(e,v,_,n),[$,H]=(0,f.GY)(M),{data:P,isLoading:O,error:j,refetchMlflowTraces:B}=(0,f.Zn)({experimentId:e,currentRunDisplayName:i,searchQuery:b,filters:w,runUuid:n,tableSort:$}),z=(0,D.C)(),{data:V,displayName:K,loading:Q}=X(e,c),J=(0,s.useMemo)((()=>({currentCount:null===P||void 0===P?void 0:P.length,logCountLoading:O,totalCount:h,maxAllowedCount:(0,U.pR)()})),[P,O,h]),{showEditTagsModalForTrace:Z,EditTagsModal:ee}=(0,A.$)({onSuccess:()=>(0,f.BL)({queryClient:E}),existingTagKeys:(0,f.d9)(P||[]),useV3Apis:!0}),ne=(0,s.useMemo)((()=>({deleteTracesAction:{deleteTraces:(e,n)=>z.mutateAsync({experimentId:e,traceRequestIds:n})},exportToEvals:{exportToEvalsInstanceEnabled:!0,getTrace:S.U},editTags:{showEditTagsModalForTrace:Z,EditTagsModal:ee}})),[z,Z,ee]),te=O||Q;return x?(0,y.Y)(G,{}):T?(0,y.Y)("div",{children:(0,y.Y)("pre",{children:String(T)})}):(0,y.FD)("div",{css:(0,a.AH)({marginTop:l.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[(0,y.Y)("div",{css:(0,a.AH)({width:"100%",padding:`${l.spacing.xs}px 0`},""),children:(0,y.Y)(Y,{experimentId:e,currentRunUuid:n,compareToRunUuid:c,setCompareToRunUuid:m,setCurrentRunUuid:r})}),(0,y.Y)(f.sG,{children:(0,y.FD)("div",{css:q,children:[(0,y.Y)(f.w_,{experimentId:e,searchQuery:b,setSearchQuery:C,filters:w,setFilters:R,assessmentInfos:g,countInfo:J,traceActions:ne,tableSort:$,setTableSort:H,allColumns:v,selectedColumns:M,setSelectedColumns:L,toggleColumns:F,traceInfos:P,tableFilterOptions:I}),te?(0,y.Y)(G,{}):j?(0,y.Y)("div",{children:(0,y.Y)("pre",{children:String(j)})}):(0,y.Y)(f.tU,{makeHtml:d,children:(0,y.Y)(f._p,{experimentId:e,currentRunDisplayName:i,compareToRunDisplayName:K,compareToRunUuid:c,getTrace:S.U,getRunColor:k,assessmentInfos:g,setFilters:R,filters:w,selectedColumns:M,allColumns:v,tableSort:$,currentTraceInfoV3:P||[],compareToTraceInfoV3:V,onTraceTagsEdit:Z})}),ee]})})]})},Q=({experimentId:e,runUuid:n,runTags:t,runDisplayName:a,setCurrentRunUuid:r})=>{const o=R(t),s=Boolean(n),{data:l,isLoading:d}=(0,f.Ie)({runUuid:n||"",artifacts:o||void 0},{disabled:!s});return d?(0,y.Y)(G,{}):!(0,i.isNil)(l)&&l.length>0?(0,y.Y)(z,{experimentId:e,runUuid:n,runDisplayName:a,data:l,runTags:t}):(0,y.Y)(K,{experimentId:e,runUuid:n,runDisplayName:a,setCurrentRunUuid:r})},G=()=>{const{theme:e}=(0,o.u)();return(0,y.Y)("div",{css:(0,a.AH)({display:"block",marginTop:e.spacing.md,height:"100%",width:"100%"},""),children:[...Array(10).keys()].map((e=>(0,y.Y)(r.ParagraphSkeleton,{label:"Loading...",seed:`s-${e}`},e)))})},X=(e,n)=>{const{data:t,isLoading:a}=(0,f.Zn)({experimentId:e,currentRunDisplayName:void 0,runUuid:n,disabled:(0,i.isNil)(n)}),{data:r,loading:o}=j({experimentIds:[e],filter:`attributes.runId = "${n}"`,disabled:(0,i.isNil)(n)});return{data:t,displayName:c.A.getRunDisplayName(null===r||void 0===r?void 0:r.info,n),loading:a||o}}}}]);
//# sourceMappingURL=7364.b91e87c1.chunk.js.map