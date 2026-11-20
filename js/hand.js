/* Stylized anatomical mesh. Pose numbers follow the UCI label convention. */
function drawPose(canvas, pose, hue = 253) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr,dpr);
  const scale = Math.min(w/164,h/205);
  const bend = pose[0], deviation = pose[1], curl = pose[2];
  const faces = [];
  function project(x,y,z, fixed) {
    if (!fixed) {
      const v = y*Math.cos(bend)-z*Math.sin(bend);
      z = y*Math.sin(bend)+z*Math.cos(bend);
      y = v;
      const u = x*Math.cos(deviation)-y*Math.sin(deviation);
      y = x*Math.sin(deviation)+y*Math.cos(deviation); x=u;
    }
    const px=x*.84+z*.54, depth=z*.84-x*.54;
    return [w*.50+scale*px,h*.69+scale*(y+depth*.14),depth];
  }
  function surface(rows, fixed=false) {
    const sides=16;
    const rings=rows.map(([x,y,z,rx,rz,angle=0])=>Array.from({length:sides},(_,i)=>{
      const a=i*2*Math.PI/sides, dz=rz*Math.sin(a);
      return project(x+rx*Math.cos(a),y+dz*Math.sin(angle),z+dz*Math.cos(angle),fixed);
    }));
    for(let j=1;j<rings.length;j++) for(let i=0;i<sides;i++){
      const k=(i+1)%sides, pts=[rings[j-1][i],rings[j-1][k],rings[j][k],rings[j][i]];
      faces.push({pts,depth:pts.reduce((s,p)=>s+p[2],0)/4});
    }
  }
  surface([[0,55,0,17,11],[0,35,0,15,10],[0,15,0,13,9],[0,0,0,14,9]],true);
  surface([[0,1,0,14,9],[0,-10,1,19,10],[0,-25,1,25,12],[0,-40,1,27,11],[0,-54,0,25,8],[0,-59,0,21,4]]);
  [-19,-6,7,19].forEach((base,i)=>{
    let x=base,y=-53+(i===3?5:0),z=0;
    const radius=[6,6.2,5.8,5][i], length=[49,56,51,39][i];
    const rows=[[x,y,z,radius,radius]];
    for(let n=1;n<=9;n++){
      const angle=.12+curl*(n<=3?.8:n<=6?1.95:2.9);
      x+=base*.005*(1-curl); y-=length/9*Math.cos(angle); z+=length/9*Math.sin(angle);
      const r=radius*(n===9?.2:1-n*.035);
      rows.push([x,y,z,r,r,angle]);
    }
    surface(rows);
  });
  const thumb=[];
  for(let j=0;j<=7;j++){
    const s=j/7;
    thumb.push([-17-27*s+curl*30*s*s,-16-38*s,4+9*s+curl*16*s,8-5*s,8-5*s]);
  }
  surface(thumb);
  faces.sort((a,b)=>a.depth-b.depth);
  ctx.lineWidth=.75;
  for(const {pts,depth} of faces){
    const light=Math.max(0,Math.min(1,(depth+33)/85));
    ctx.fillStyle='hsl('+hue+' 25% '+(27+light*38)+'%)';
    ctx.strokeStyle='hsla('+hue+',80%,88%,.65)';
    ctx.beginPath(); pts.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));
    ctx.closePath();ctx.fill();ctx.stroke();
  }
}
