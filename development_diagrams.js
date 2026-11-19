"use strict";

const featureLabels = ["RMS", "MAV", "WL"];
const numeric = value => value.toPrecision(4);

function calculationCards(ctx, entries, progress, left, right) {
  const width = (right-left)/entries.length;
  entries.forEach(([name,value],i)=>{
    const x=left+i*width, fade=Math.max(0,Math.min(1,progress*3-i*.42));
    box(ctx,x,258,width-8,63,"#253144");
    ctx.globalAlpha=fade;
    text(ctx,x+10,275,name,muted,14);
    text(ctx,x+10,301,numeric(value),i===entries.length-1?mint:bright,20);
    ctx.globalAlpha=1;
  });
}

function drawTrainingMatrix(ctx,data,p,{matrix:mode}){
  const matrix=data.matrix, standardized=mode==="scaled", names=matrix.columns.slice(0,4);
  const values=standardized?matrix.z:matrix.rows;
  const columns=[160,285,410,535];
  text(ctx,23,26,(standardized?"标准化训练矩阵 Z_train":"训练特征矩阵 X_train")+" · "+matrix.shape.join(" × "),bright,21);
  text(ctx,73,72,"y_train",muted,15,"center");
  names.forEach((name,i)=>text(ctx,columns[i]+52,72,name,muted,15,"center"));
  matrix.labels.forEach((label,i)=>{
    const y=92+i*30, phase=Math.max(0,Math.min(1,p*2.25-i*.18));
    if(standardized || p>i*.11){
      box(ctx,25,y,96,24,"#26374a"); text(ctx,73,y+12,gesture(label),colors[label-1],15,"center");
    }
    columns.forEach((x,j)=>{
      const cell=values[i][j];
      const offset=standardized?0:(1-phase)*(14+i*5);
      const converted=standardized && p*1.35>j*.30;
      ctx.globalAlpha=standardized?1:phase;
      box(ctx,x+offset,y,105,24,converted?"#30483d":"#273548");
      text(ctx,x+offset+52,y+12,numeric(converted||!standardized?cell:matrix.rows[i][j]),converted?mint:bright,15,"center");
      ctx.globalAlpha=1;
    });
  });
  text(ctx,24,294,standardized?"z = (x − μ) / σ":"展示 6 个窗口 × 4 列 · 完整矩阵含全部 24 列",muted,15);
  if(standardized){
    const raw=matrix.rows[0][0], mean=matrix.mean[0], scale=matrix.scale[0];
    text(ctx,214,294,"第一格  ("+numeric(raw)+" − "+numeric(mean)+") / "+numeric(scale)+" → "+numeric(matrix.z[0][0]),mint,15);
  }
}

function drawLogisticDetails(ctx,data,p,state){
  if(state.logistic==="curve"){
    drawLogistic(ctx,data,p,state);
    return;
  }
  const curve=data.curve;
  const probabilities=state.logistic==="softmax";
  const values=probabilities?curve.probabilities:curve.logits;
  const winner=curve.prediction-1;
  const span=probabilities?1:Math.max(...values.map(Math.abs),1e-9);
  const zero=probabilities?120:390;
  const width=probabilities?540:265;
  text(ctx,24,27,probabilities?"Softmax · exp(sᵢ) / Σ exp(sⱼ)":"每类分数 · sᵢ = Σ (wᵢⱼ × zⱼ) + bᵢ",bright,20);
  values.forEach((value,i)=>{
    const y=57+i*34, length=value/span*width*p;
    text(ctx,27,y+10,gestures[i],i===winner?mint:muted,16);
    box(ctx,120,y,540,19,"#233146");
    box(ctx,zero+Math.min(0,length),y,Math.abs(length),19,i===winner?mint:colors[i]);
    text(ctx,733,y+10,probabilities?percent(value):value.toFixed(3),i===winner?mint:bright,16,"right");
  });
  if(!probabilities){
    line(ctx,[[zero,52],[zero,250]],"#768397");
    const terms=curve.contributions[winner];
    const top=terms.map((_,i)=>i).sort((a,b)=>Math.abs(terms[b])-Math.abs(terms[a])).slice(0,2);
    const remainder=terms.reduce((sum,v,i)=>sum+(top.includes(i)?0:v),0);
    text(ctx,24,280,gesture(winner+1)+"得分拆解   "+top.map(i=>curve.columns[i]+" "+terms[i].toFixed(2)).join("   ·   "),muted,15);
    text(ctx,24,308,"其余特征 "+remainder.toFixed(2)+"   ·   偏置 "+curve.intercept[winner].toFixed(2)+"   →   总分 "+curve.logits[winner].toFixed(3),bright,15);
  }else{
    text(ctx,24,282,"六类概率之和 = 100%  ·  最大值对应预测动作",muted,16);
    text(ctx,24,309,"预测 "+gesture(winner+1)+"   "+percent(values[winner])+"     /     标注 "+gesture(data.sample.label),mint,18);
  }
}

function drawArtifacts(ctx,data,p,{artifact,artifactMetric,artifactClass}){
  const matrix=data.matrix, metric=featureLabels[artifactMetric];
  if(artifact==="tree"){
    drawForest(ctx,data,p,{depth:3});
    return;
  }
  const columns=Array.from({length:8},(_,ch)=>ch*3+artifactMetric);
  if(artifact==="relation"){
    text(ctx,26,25,metric+" · 各类别的平均标准化特征",bright,19);
    const limit=Math.max(...matrix.class_mean.flatMap(row=>columns.map(i=>Math.abs(row[i]))),1e-9);
    columns.forEach((_,j)=>text(ctx,126+j*78,62,"CH "+(j+1),muted,14,"center"));
    matrix.class_mean.forEach((row,i)=>{
      const y=79+i*32;
      text(ctx,65,y+13,gestures[i],colors[i],16,"right");
      columns.forEach((j,ch)=>{
        const value=row[j], intensity=Math.abs(value)/limit*p;
        const shade=value<0?"145,152,232":"125,208,150";
        box(ctx,91+ch*78,y,70,26,"#273449");
        box(ctx,91+ch*78,y,70,26,`rgba(${shade},${intensity*.68})`);
        text(ctx,126+ch*78,y+13,value.toFixed(2),bright,14,"center");
      });
    });
    text(ctx,26,299,"紫色 · 低于训练均值      绿色 · 高于训练均值      数值为各类平均 z",muted,15);
    return;
  }
  if(artifact==="scale"){
    text(ctx,26,25,metric+" · 训练集拟合出的 μ 与 σ",bright,19);
    columns.forEach((index,ch)=>{
      const x=27+(ch%4)*181,y=59+Math.floor(ch/4)*106;
      box(ctx,x,y,171,92,"#26354a");
      ctx.globalAlpha=.4+.6*p;
      text(ctx,x+12,y+20,"CH "+(ch+1),mint,16);
      text(ctx,x+12,y+47,"μ  "+numeric(matrix.mean[index]),bright,15);
      text(ctx,x+12,y+73,"σ  "+numeric(matrix.scale[index]),bright,15);
      ctx.globalAlpha=1;
    });
    text(ctx,26,299,"新窗直接套用 z = (x − μ) / σ",muted,15);
    return;
  }
  const curve=data.curve;
  text(ctx,26,26,"逻辑回归 · "+gestures[artifactClass]+"的权重与偏置",bright,19);
  const weights=curve.weights[artifactClass].slice(0,8);
  weights.forEach((value,i)=>{
    const x=27+(i%4)*181,y=59+Math.floor(i/4)*74;
    box(ctx,x,y,171,64,"#26354a");
    text(ctx,x+12,y+18,curve.columns[i],muted,13);
    text(ctx,x+12,y+44,numeric(value*p),value>=0?mint:"#c4b5fa",18);
  });
  text(ctx,26,220,"偏置 b = "+numeric(curve.intercept[artifactClass])+" · "+curve.weights[artifactClass].length+" 项权重 · 图中展示前 8 项",muted,15);
  text(ctx,26,252,"逻辑回归 · "+data.best.logistic.parameter+"      kNN · "+data.best.knn.parameter+"，保留训练样本",bright,15);
  const {trees,max_depth,criterion}=data.forest_params;
  text(ctx,26,284,"随机森林 · "+trees+" 棵树 · 深度上限 "+(max_depth??"未指定")+" · "+criterion,bright,15);
}
