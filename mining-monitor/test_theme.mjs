import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

export async function runThemeTests(){
  const source=await readFile(new URL('./theme.js',import.meta.url),'utf8');
  const passed=[];
  function boot(saved=null,dark=false){
    const elements={},properties={},storage=new Map(saved?[['oro-monitor-appearance',saved]]:[]);
    const system={matches:dark,addEventListener(type,handler){this.change=handler;}};
    let ready;
    const document={documentElement:{dataset:{},style:{setProperty(key,value){properties[key]=value;}}},
      addEventListener(type,handler){if(type==='DOMContentLoaded')ready=handler;},
      getElementById(id){return elements[id] ||= {value:'',textContent:'',open:false,
        showModal(){this.open=true;},close(){this.open=false;},setCustomValidity(value){this.validity=value;}};}};
    vm.runInNewContext(source,{document,matchMedia:()=>system,localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)}});
    ready();elements['appearance-open'].onclick();
    return {elements,properties,storage,system,root:document.documentElement};
  }
  let app=boot();assert.equal(app.root.dataset.theme,'light');assert.equal(app.properties['--paper'],'#e5ebf2');assert.equal(app.elements['appearance-current'].textContent,'Light · Auto');passed.push('default light palette and visible system-selected mode');
  app.system.matches=true;app.system.change();assert.equal(app.root.dataset.theme,'dark');assert.equal(app.elements['appearance-current'].textContent,'Dark · Auto');passed.push('system mode follows system changes');
  app.elements['theme-mode'].onchange({target:{value:'light'}});app.system.change();assert.equal(app.root.dataset.theme,'light');assert.equal(app.elements['appearance-current'].textContent,'Light');passed.push('explicit light mode overrides system');
  app.elements['theme-mode'].onchange({target:{value:'dark'}});assert.equal(app.root.dataset.theme,'dark');passed.push('explicit dark mode');
  const field=app.elements['hex-accent'];field.value='#aabbcc';field.oninput({target:field});assert.equal(app.properties['--green'],'#aabbcc');assert.equal(app.elements['theme-accent'].value,'#aabbcc');passed.push('custom hex color applies immediately');
  const saved=app.storage.get('oro-monitor-appearance');let resumed=boot(saved);assert.equal(resumed.root.dataset.theme,'dark');assert.equal(resumed.properties['--green'],'#aabbcc');passed.push('mode and colors persist across reload');
  resumed.elements['theme-mode'].onchange({target:{value:'light'}});assert.equal(resumed.properties['--green'],'#355d8c');resumed.elements['theme-mode'].onchange({target:{value:'dark'}});assert.equal(resumed.properties['--green'],'#aabbcc');passed.push('light and dark palettes remain independent');
  const invalid=resumed.elements['hex-accent'];invalid.value='invalid';invalid.oninput({target:invalid});assert.equal(resumed.properties['--green'],'#aabbcc');assert.ok(invalid.validity);passed.push('invalid hex values are rejected');
  resumed.elements['theme-reset'].onclick();assert.equal(resumed.properties['--green'],'#a6bfdf');passed.push('reset restores selected palette');
  const corrupt=boot('{broken');assert.equal(corrupt.root.dataset.theme,'light');passed.push('corrupt preferences recover safely');
  const injected=boot(JSON.stringify({mode:'dark',palettes:{dark:{accent:'url(https://invalid.example)'}}}));assert.equal(injected.properties['--green'],'#a6bfdf');passed.push('stored colors are validated before applying CSS');
  const luminance=color=>{
    const rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16)/255);
    return rgb.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
  };
  const contrast=(a,b)=>(Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);
  for(const dark of [false,true]){
    const p=boot(null,dark).properties;
    for(const background of ['--paper','--surface','--surface-alt']){
      assert.ok(contrast(p['--ink'],p[background])>=4.5,'body text remains readable');
      assert.ok(contrast(p['--muted'],p[background])>=4.5,'secondary text remains readable');
    }
    assert.ok(contrast(p['--sidebar-ink'],p['--sidebar'])>=4.5);
    assert.ok(contrast(p['--sidebar-muted'],p['--sidebar'])>=4.5);
    assert.ok(contrast(p['--on-accent'],p['--green'])>=4.5);
    passed.push((dark?'dark':'light')+' body, secondary, navigation and button text have readable contrast');
    for(const state of ['success','warning','danger'])assert.ok(contrast(p['--'+state],p['--'+state+'-soft'])>=4.5,state+' label stays legible');
    assert.notEqual(p['--success-soft'],p['--warning-soft']);assert.notEqual(p['--warning-soft'],p['--danger-soft']);
    passed.push((dark?'dark':'light')+' status labels have distinct subdued fills and readable text');
    assert.ok(contrast(p['--surface'],p['--paper'])>=1.1,'panels separate from the workspace');
    assert.ok(contrast(p['--line-strong'],p['--surface'])>contrast(p['--line-soft'],p['--surface']),'outer edges are clearer than internal dividers');
    assert.ok(contrast(p['--line-strong'],p['--surface'])>=1.6,'outer borders remain visible');
    assert.ok(contrast(p['--control-line'],p['--control-surface'])>=1.8,'inputs have visible boundaries');
    passed.push((dark?'dark':'light')+' panels, dividers and input outlines have distinct boundary levels');
  }
  return passed;
}
