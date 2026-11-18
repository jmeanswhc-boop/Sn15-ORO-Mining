'use strict';
(() => {
  const defaults={
    light:{background:'#e5ebf2',surface:'#f6f8fb',text:'#304155',accent:'#355d8c',sidebar:'#dbe3ed',border:'#b6c3d2'},
    dark:{background:'#161f2a',surface:'#243142',text:'#c7d2e0',accent:'#a6bfdf',sidebar:'#1b2735',border:'#53667d'}
  };
  let settings={mode:'system',palettes:{}};
  try {const saved=JSON.parse(localStorage.getItem('oro-monitor-appearance'));if(saved && ['light','dark','system'].includes(saved.mode)) settings={mode:saved.mode,palettes:saved.palettes && typeof saved.palettes==='object' ? saved.palettes : {}};} catch {}
  const system=matchMedia('(prefers-color-scheme: dark)');
  const valid=value=>/^#[0-9a-f]{6}$/i.test(value || '');
  const rgb=value=>[1,3,5].map(i=>parseInt(value.slice(i,i+2),16));
  const mix=(a,b,weight)=>'#'+rgb(a).map((n,i)=>Math.round(n*weight+rgb(b)[i]*(1-weight)).toString(16).padStart(2,'0')).join('');
  const luminance=color=>rgb(color).map(n=>{const v=n/255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
  const contrast=(a,b)=>(Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);
  const on=color=>{
    const soft=['#253446','#edf2f8'].sort((a,b)=>contrast(b,color)-contrast(a,color));
    return contrast(soft[0],color)>=4.5?soft[0]:(luminance(color)>.179?'#000000':'#ffffff');
  };
  function effective(){return settings.mode==='system'?(system.matches?'dark':'light'):settings.mode;}
  function palette(){const mode=effective(), result={...defaults[mode]};for(const key of Object.keys(result)){const value=settings.palettes[mode]?.[key];if(valid(value)) result[key]=value;}return result;}
  function apply(){
    const p=palette(), root=document.documentElement;
    root.dataset.theme=effective();
    const sidebarInk=contrast(p.text,p.sidebar)>=4.5?p.text:on(p.sidebar);
    const dark=effective()==='dark';
    const states=dark?{success:'#b1d6c3',warning:'#e2c58e',danger:'#e0b2b6'}:{success:'#305f4c',warning:'#76541f',danger:'#88474e'};
    const variables={paper:p.background,surface:p.surface,ink:p.text,green:p.accent,line:p.border,sidebar:p.sidebar,
      muted:mix(p.text,p.surface,.8),'surface-alt':mix(p.surface,p.background,.8),hover:mix(p.accent,p.surface,.1),
      'surface-heading':mix(p.surface,p.background,.48),
      'line-strong':mix(p.border,p.text,dark ? .78 : .85),'line-soft':mix(p.border,p.surface,.55),
      'control-line':mix(p.border,p.text,dark ? .72 : .78),'control-surface':p.surface,
      'panel-shadow':dark?'0 2px 5px #080f1726':'0 2px 4px #243d5a0d',
      lime:p.accent,'on-accent':on(p.accent),'sidebar-ink':sidebarInk,'sidebar-muted':mix(sidebarInk,p.sidebar,.82),
      'sidebar-hover':mix(p.accent,p.sidebar,.12),focus:p.accent,
      'accent-soft':mix(p.accent,p.surface,.1),...states,
      'success-soft':mix(states.success,p.surface,.1),'warning-soft':mix(states.warning,p.surface,.1),
      'danger-soft':mix(states.danger,p.surface,.1)};
    for(const [key,value] of Object.entries(variables)) root.style.setProperty('--'+key,value);
    const modeLabel=document.getElementById('appearance-current');
    const custom=Object.values(settings.palettes[effective()] || {}).some(valid);
    if(modeLabel)modeLabel.textContent=`${dark?'Dark':'Light'}${settings.mode==='system'?' · Auto':''}${custom?' · Custom':''}`;
  }
  function save(){try {localStorage.setItem('oro-monitor-appearance',JSON.stringify(settings));} catch {}apply();}
  function controls(){
    const p=palette();
    document.getElementById('theme-mode').value=settings.mode;
    for(const [name,color] of Object.entries(p)){document.getElementById('theme-'+name).value=color;document.getElementById('hex-'+name).value=color;document.getElementById('hex-'+name).setCustomValidity('');}
    const a=luminance(p.text),b=luminance(p.background),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
    document.getElementById('theme-contrast').textContent=`Text/background contrast: ${ratio.toFixed(1)}:1${ratio<4.5?' · Low contrast; adjust text or background.':''}`;
  }
  apply();
  system.addEventListener('change',()=>{if(settings.mode==='system'){apply();if(document.getElementById('appearance')?.open) controls();}});
  document.addEventListener('DOMContentLoaded',()=>{
    apply();
    const dialog=document.getElementById('appearance');
    document.getElementById('appearance-open').onclick=()=>{controls();dialog.showModal();};
    document.getElementById('appearance-close').onclick=()=>dialog.close();
    document.getElementById('theme-mode').onchange=event=>{settings.mode=event.target.value;save();controls();};
    for(const name of Object.keys(defaults.light)) {
      const change=value=>{const mode=effective();if(!settings.palettes[mode] || typeof settings.palettes[mode]!=='object')settings.palettes[mode]={};settings.palettes[mode][name]=value;save();controls();};
      document.getElementById('theme-'+name).oninput=event=>change(event.target.value);
      document.getElementById('hex-'+name).oninput=event=>{const ok=valid(event.target.value);event.target.setCustomValidity(ok?'':'Use a six-digit hex color such as #23594c');if(ok)change(event.target.value);};
    }
    document.getElementById('theme-reset').onclick=()=>{delete settings.palettes[effective()];save();controls();};
  });
})();
