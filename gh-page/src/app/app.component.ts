import { Component, OnInit, ElementRef, ViewChild, HostListener } from '@angular/core';
import { StepperSelectionEvent } from '@angular/cdk/stepper';
// import * as vtree from '../assets/vtree.min.js';

declare var vtree: any

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent implements OnInit {
  @ViewChild('codeView', {static: false}) codeView: ElementRef;
  @ViewChild('fileInput', {static: false}) fileView: ElementRef;

  code: string = `# demonstration of preprocessing and normalizing imports
import pandas as pd
a = pd.DataFrame()

# demonstration of normalization of chained calls
pd.DataFrame([[1, 2, 3], [4, 5, 6]]).sum().mean()

# demonstration of UDF, UDV, built-in/package methods differentiation
def c(d):
	return d + 1
# e.g. we called b = function_returner()
b()
c()
sum(1, 2, 3)`
  vt: any
  imports: Map<string, string> = new Map<string, string>();
  ngOnInit(): void {
  }

  processTab(e: KeyboardEvent) {
    if (e.key === 'Tab') {
      let start = this.codeView.nativeElement.selectionStart, end = this.codeView.nativeElement.selectionEnd
      this.code = this.code.substring(0, start) + '\t' + this.code.substring(end)
      this.codeView.nativeElement.selectionStart = this.codeView.nativeElement.selectionEnd = start + 1
      return false
    }
  }

  async sendCode() {
    this.vt = vtree(document.getElementById('container')).conf('maxNameLen', 32).conf('maxValueLen', 32)
    this.vt.mode(this.vt.MODE_PYTHON_AST).conf('showArrayNode', false)
    document.querySelector('svg').setAttribute('width', <string><unknown>(window.innerWidth * .9))
    let request = await fetch('http://demo.codehonestly.com/ast2json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `pysrc=${encodeURIComponent(this.code)}&normalization=true&ctx=false`
    })
    let response = await request.json()
    this.vt.data(response['graph']).update()

    this.imports.clear()
    for (let [key, item] of Object.entries(response['imports'])) {
      this.imports.set(key, <string>item)
    }
  }

  selectionChange(e: StepperSelectionEvent) {
    if (e.selectedIndex === 2) {
      this.sendCode()
    }
  }
}
