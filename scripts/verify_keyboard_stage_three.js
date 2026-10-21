async page => {
 const check=(x,m)=>{if(!x)throw new Error(m)};
 await page.setViewportSize({width:1440,height:900});
 await page.goto('http://127.0.0.1:8518/');
 const predict=page.getByRole('button',{name:'Prediksi kategori',exact:true});
 await predict.waitFor();
 await page.waitForFunction(()=>document.getElementById('isi-profil')?.getAttribute('tabindex')==='-1');
 const order=await page.getByRole('textbox').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('aria-label')));
 check(order[0]==='Usia (tahun)'&&order[1]==='Pendapatan tahunan (unit uang dataset)'&&order[2]==='Pendapatan bersih bulanan (unit uang dataset)','Keyboard and visual input order diverge');
 const tabSequence=[];
 for(let i=0;i<12;i++){
  await page.keyboard.press('Tab');
  tabSequence.push(await page.evaluate(()=>({tag:document.activeElement.tagName,name:document.activeElement.getAttribute('aria-label')||document.activeElement.innerText})));
  if(await page.locator('.skip-link').evaluate(n=>n===document.activeElement)) break;
 }
 check(await page.locator('.skip-link').evaluate(n=>n===document.activeElement),'Skip link not keyboard reachable');
 await page.keyboard.press('Enter');
 await page.waitForFunction(()=>document.activeElement.id==='isi-profil');
 const age=page.getByRole('textbox',{name:'Usia (tahun)',exact:true});
 await age.focus();await page.keyboard.press('Tab');
 const ring=await page.evaluate(()=>{const s=getComputedStyle(document.activeElement);return {style:s.outlineStyle,width:s.outlineWidth,color:s.outlineColor}});
 check(ring.style==='solid'&&parseFloat(ring.width)>=3,'Keyboard focus not visible');
 await predict.focus();await page.keyboard.press('Enter');
 await page.getByText(/Kategori model:/).waitFor();
 await page.waitForFunction(()=>document.activeElement.id==='hasil-prediksi');
 await page.getByText('Lihat benchmark dan kesalahan per kelas',{exact:true}).click();
 // Measure the one paragraph axe could not resolve when partly under the fixed toolbar.
 const paragraph=page.getByText('Konteks untuk menilai model secara keseluruhan, bukan tingkat keyakinan profil yang sedang kamu isi.',{exact:true});
 const contrast=await paragraph.evaluate(el=>{
  const c=getComputedStyle(el).color;
  let n=el,b='rgba(0, 0, 0, 0)';while(n&&b==='rgba(0, 0, 0, 0)'){b=getComputedStyle(n).backgroundColor;n=n.parentElement}
  const luminance=s=>{const rgb=s.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4);return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]};
  const a=luminance(c),d=luminance(b);return {foreground:c,background:b,ratio:(Math.max(a,d)+.05)/(Math.min(a,d)+.05)};
 });
 check(contrast.ratio>=4.5,'Unresolved paragraph contrast fails');
 await page.setViewportSize({width:320,height:900});
 const matrix=page.locator('.matrix');await matrix.scrollIntoViewIfNeeded();
 const tableFit=await matrix.evaluate(n=>({table:n.getBoundingClientRect().width,region:n.parentElement.clientWidth,scroll:n.parentElement.scrollWidth}));
 check(tableFit.scroll<=tableFit.region+1,'Mobile matrix still needs horizontal scrolling');
 await matrix.screenshot({path:'docs/screenshots/stage-three-320-matrix.png'});
 await page.setViewportSize({width:720,height:900});
 await page.addStyleTag({content:'html{font-size:200% !important}'});
 check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'200% text causes horizontal overflow');
 await page.emulateMedia({reducedMotion:'reduce'});
 await predict.focus();await page.keyboard.press('Enter');
 await page.getByText(/Kategori model:/).waitFor();
 await page.waitForFunction(()=>document.activeElement.id==='hasil-prediksi');
 check(await page.evaluate(()=>getComputedStyle(document.documentElement).scrollBehavior)==='auto','Reduced motion not respected');
 return {keyboardOnlyPrediction:true,skipLink:true,tabSequence,inputOrder:order,focusRing:ring,manualContrast:contrast,mobileMatrixFit:tableFit,text200PercentNoOverflow:true,reducedMotion:true};
}
