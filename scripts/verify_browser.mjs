import { chromium } from 'file:///C:/Users/steve/Documents/Steven%20Gerald%20Cahyadi/FeedingTomorrow/node_modules/@playwright/test/index.mjs'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import assert from 'node:assert/strict'
const meta=JSON.parse(await readFile('artifacts/metadata.json','utf8'))
const tag=process.env.QA_TAG || 'stage-one'
assert.match(tag,/^[a-z0-9-]+$/)
await mkdir('docs/screenshots',{recursive:true});await mkdir('docs/verification',{recursive:true})
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true})
const rows=[]
for(const width of [1440,390,320]){
 const page=await browser.newPage({viewport:{width,height:900}}),errors=[]
 page.on('pageerror',e=>errors.push(e.message))
 await page.goto('http://127.0.0.1:8518/')
 await page.getByRole('heading',{name:'Klasifikasi kategori kredit',exact:true}).waitFor()
 const predict=page.getByRole('button',{name:'Prediksi kategori',exact:true})
 await predict.waitFor()
 const money=page.getByRole('spinbutton',{name:'Pendapatan tahunan (unit uang dataset)',exact:true})
 assert.equal(Number(await money.inputValue()),meta.defaults.Annual_Income)
 assert.equal(Number(await money.getAttribute('max')),meta.supported_ranges.Annual_Income.max)
 assert.equal(await page.getByRole('spinbutton').count(),17)
 await page.screenshot({path:`docs/screenshots/${tag}-${width}-form.png`})
 await predict.click()
 const result=page.getByText(/Kategori model:/)
 await result.waitFor();await result.scrollIntoViewIfNeeded()
 assert.match(await result.innerText(),/Good|Standard|Poor/)
 await page.screenshot({path:`docs/screenshots/${tag}-${width}-result.png`})
 const age=page.getByRole('spinbutton',{name:'Usia (tahun)',exact:true})
 await age.fill(String(meta.defaults.Age+1));await age.press('Enter')
 await page.waitForFunction(()=>!document.body.innerText.includes('Kategori model:'))
 await page.getByRole('button',{name:'Muat contoh nilai dataset',exact:true}).click()
 await page.waitForFunction(value=>document.querySelector('input[aria-label="Usia (tahun)"]')?.value===String(value),meta.defaults.Age)
 await page.getByText('AWS SageMaker',{exact:true}).click()
 await predict.click()
 await page.getByText('Endpoint AWS belum diatur. Pilih mode Lokal atau atur endpoint di server.',{exact:true}).waitFor()
 assert.equal(await page.getByText(/Kategori model:/).count(),0)
 await page.getByText('Lokal',{exact:true}).click();await predict.click();await result.waitFor()
 if(tag==='stage-two'){
  await page.getByText('Tentang input dan model',{exact:true}).click()
  await page.getByText(/Holdout nasabah: accuracy 68,61%/).waitFor()
  await page.getByText(/Kontrak aplikasi menerima 3.067/).waitFor()
  assert.match(await page.locator('body').innerText(),new RegExp(meta.model_version))
  await page.getByText(/Holdout nasabah: accuracy 68,61%/).scrollIntoViewIfNeeded()
  await page.screenshot({path:`docs/screenshots/${tag}-${width}-evaluation.png`})
 }
 await page.addScriptTag({path:'../FeedingTomorrow/node_modules/axe-core/axe.min.js'})
 const violations=await page.evaluate(async()=> (await window.axe.run()).violations.map(x=>({id:x.id,impact:x.impact,count:x.nodes.length})))
 if(violations.some(x=>['color-contrast','heading-order'].includes(x.id))) {
  console.log(JSON.stringify(await page.evaluate(async()=> (await window.axe.run()).violations.filter(x=>['color-contrast','heading-order'].includes(x.id)).map(x=>({id:x.id,nodes:x.nodes.map(n=>({html:n.html,summary:n.failureSummary}))}))),null,2))
  await browser.close();throw new Error('Contrast or heading regression remains')
 }
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)
 assert.equal(overflow,false);assert.deepEqual(errors,[])
 rows.push({width,version:browser.version(),inputCount:18,localPrediction:true,editClearsResult:true,resetRestoresExample:true,missingAWSEndpointIsTechnicalError:true,localRecovery:true,overflow,errors,axe:violations})
 await page.close()
}
await browser.close()
await writeFile(`docs/verification/browser-${tag}.json`,JSON.stringify({origin:'http://127.0.0.1:8518',modelVersion:meta.model_version,cases:rows},null,2)+'\n')
console.log(JSON.stringify(rows,null,2))
