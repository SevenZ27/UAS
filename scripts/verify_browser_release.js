async page => {
 const check=(ok,message)=>{if(!ok) throw new Error(message)};
 const browser=page.context().browser();
 for(const old of browser.contexts()) if(old!==page.context()) await old.close();
 const cases=[];
 for(const width of [1440,900,390,320]) {
  const context=await browser.newContext({viewport:{width,height:900},isMobile:width<500,hasTouch:width<500,locale:'id-ID'});
  const p=await context.newPage(); const errors=[];
  p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://127.0.0.1:8518/');
  const predict=p.getByRole('button',{name:'Prediksi kategori',exact:true});
  await predict.waitFor();
  await p.waitForFunction(()=>document.querySelector('[data-testid="stMain"]')?.getAttribute('role')==='main');
  check(await p.getByRole('textbox').count()===17,'Expected 17 numeric text fields');
  check(await p.locator('.category-list [aria-current]').count()===0,'Empty result implies a label');
  await p.screenshot({path:`docs/screenshots/release-${width}-empty.png`});
  const activate=async locator=>width<500?locator.tap():locator.click();
  await activate(predict);
  const result=p.getByText(/Kategori model:/);
  await result.waitFor();
  await p.waitForFunction(()=>document.activeElement.id==='hasil-prediksi');
  check(await p.locator('.category-list [aria-current="true"]').count()===1,'No category selection');
  if(width===1440){
   const [download]=await Promise.all([p.waitForEvent('download'),p.getByRole('button',{name:'Unduh ringkasan JSON'}).click()]);
   await download.saveAs('qa/browser-release-download.json');
  }
  await p.screenshot({path:`docs/screenshots/release-${width}-result.png`});
  const age=p.getByRole('textbox',{name:'Usia (tahun)',exact:true});
  await age.fill('34');await age.press('Enter');
  await p.waitForFunction(()=>!document.body.innerText.includes('Kategori model:'));
  check(await age.evaluate(x=>x===document.activeElement),'Edit stole keyboard focus');
  await age.fill('');await age.press('Enter');await activate(predict);
  await p.getByText(/Usia:.*wajib/).waitFor();
  await p.waitForFunction(()=>document.activeElement.id==='prediction-error');
  check(await result.count()===0,'Validation failure produced category');
  await p.getByRole('button',{name:'Muat contoh nilai dataset',exact:true}).click();
  await p.waitForFunction(()=>document.querySelector('input[aria-label="Usia (tahun)"]').value==='33');
  // Test the native typed out-of-range state without accepting a stale profile.
  await age.fill('999');await age.press('Enter');
  await activate(predict);
  await p.getByText(/Usia: Nilai maksimum/).waitFor();
  check(await result.count()===0,'Out-of-range value produced a stale category');
  await p.getByRole('button',{name:'Muat contoh nilai dataset',exact:true}).click();
  await p.waitForFunction(()=>document.querySelector('input[aria-label="Usia (tahun)"]').value==='33');
  await age.fill('70');await age.press('Enter');await activate(predict);await result.waitFor();
  await p.getByText(/keandalannya untuk profil ini belum teruji/).waitFor();
  await p.getByText('Rincian nilai di luar rentang',{exact:true}).click();
  await p.getByText(/Usia: 70/).waitFor();
  if(width===1440){
   const [download]=await Promise.all([p.waitForEvent('download'),p.getByRole('button',{name:'Unduh ringkasan JSON'}).click()]);await download.saveAs('qa/browser-release-outside.json');
  }
  await p.screenshot({path:`docs/screenshots/release-${width}-outside.png`});
  await p.getByRole('button',{name:'Muat contoh nilai dataset',exact:true}).click();
  await p.waitForFunction(()=>document.querySelector('input[aria-label="Usia (tahun)"]').value==='33');
  const salary=p.getByRole('textbox',{name:'Pendapatan bersih bulanan',exact:true});
  await salary.fill('');await salary.press('Enter');await activate(predict);await result.waitFor();
  await p.getByText('Nilai yang tidak diketahui diisi menggunakan median data training.',{exact:true}).waitFor();
  await p.getByText('Pengaturan sumber prediksi',{exact:true}).click();
  await p.getByText('AWS SageMaker',{exact:true}).click();
  await p.waitForFunction(()=>!document.body.innerText.includes('Kategori model:'));
  await activate(predict);
  await p.getByText('Endpoint AWS belum diatur. Pilih mode Lokal atau atur endpoint di server.',{exact:true}).waitFor();
  check(await result.count()===0,'AWS failure produced category');
  await p.getByText('Lokal',{exact:true}).click();await activate(predict);await result.waitFor();
  await p.getByText('Input yang dipakai dan versi model',{exact:true}).click();
  await p.getByText('Lihat benchmark dan kesalahan per kelas',{exact:true}).click();
  await p.getByRole('table',{name:'Confusion matrix holdout; persen per kelas aktual'}).waitFor();
  await p.getByRole('table',{name:'Confusion matrix holdout; persen per kelas aktual'}).scrollIntoViewIfNeeded();
  await p.screenshot({path:`docs/screenshots/release-${width}-evaluation.png`});
  await p.addScriptTag({path:'node_modules/axe-core/axe.min.js'});
  const axe=await p.evaluate(async()=>{
   const r=await window.axe.run();return {violations:r.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({html:n.html,summary:n.failureSummary}))})),incomplete:r.incomplete.map(v=>({id:v.id,count:v.nodes.length}))};
  });
  const sizes=await p.locator('[data-testid="stTextInput"] input').evaluateAll(nodes=>nodes.map(n=>({height:n.getBoundingClientRect().height,name:n.getAttribute('aria-label'),mode:n.getAttribute('inputmode')})));
  check(axe.violations.length===0,'Axe accessibility violations remain');
  const overflow=await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  const headings=await p.locator('h1').count();
  check(!overflow,'Horizontal page overflow');check(headings===1,'Page needs one h1');check(errors.length===0,'JavaScript page error');
  check(sizes.every(x=>x.height>=43.9&&x.name&&x.mode),'Small or unnamed numeric fields');
  if(width===1440){
   await p.emulateMedia({reducedMotion:'reduce'});
   await p.getByRole('button',{name:'Muat contoh nilai dataset',exact:true}).click();
   await p.keyboard.press('Tab');
   await predict.focus();await p.keyboard.press('Enter');await result.waitFor();
   await p.waitForFunction(()=>document.activeElement.id==='hasil-prediksi');
   check(await p.evaluate(()=>getComputedStyle(document.documentElement).scrollBehavior)==='auto','Reduced motion ignored');
  }
  cases.push({width,touchEmulated:width<500,localPrediction:true,resultFocus:true,editClearsResult:true,editPreservesFocus:true,namedNumericFields:true,missingAgeError:true,missingFeatureWarning:true,backendErrorAndRecovery:true,invalidAgeRejected:true,outsideTrainingRangeResultWithWarning:true,overflow,errors,axe});
  await context.close();
 }
 return {origin:'http://127.0.0.1:8518/',tool:'Playwright CLI',cases};
}
