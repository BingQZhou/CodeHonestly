import { Component, ElementRef, ViewChild, AfterViewInit, ViewChildren, QueryList } from '@angular/core';
// import { StepperSelectionEvent } from '@angular/cdk/stepper';
// import * as vtree from '../assets/vtree.min.js';

declare var vtree: any

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent implements AfterViewInit {
  // @ViewChild('codeView') codeView: ElementRef
  // @ViewChild('codeView2') codeView2: ElementRef
  @ViewChildren('codeView') codeViews: QueryList<ElementRef>

  code: string[] = [`# demonstration of preprocessing and normalizing imports
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
sum(1, 2, 3)`, '']
  vt: any
  imports: Map<string, string> = new Map<string, string>()

  visualizeMode: boolean = true

  ngAfterViewInit(): void {
    this.vt = vtree(document.getElementById('container')).conf('maxNameLen', 32).conf('maxValueLen', 32)
    this.vt.mode(this.vt.MODE_PYTHON_AST).conf('showArrayNode', false)
    document.querySelector('svg').setAttribute('width', '90vw')
    document.querySelector('svg').setAttribute('height', '90vh')
    document.querySelector('svg').style.boxShadow = 'none'
    this.vt.height = document.querySelector('svg').clientHeight
    this.vt.width = document.querySelector('svg').clientWidth
    this.sendCode()
  }

  processTab(e: KeyboardEvent, index=0): boolean {
    let asArray: Array<ElementRef> = this.codeViews.toArray()
    if (e.key === 'Tab') {
      let start: number = asArray[index].nativeElement.selectionStart, end: number = asArray[index].nativeElement.selectionEnd
      this.code[index] = this.code[index].substring(0, start) + '\t' + this.code[index].substring(end)
      setTimeout(function(): void {
        asArray[index].nativeElement.selectionStart = asArray[index].nativeElement.selectionEnd = start + 1
      }.bind(this), 0)
      return false
    }
  }

  processFile(e: any, index=0): void {
    let fr: FileReader = new FileReader()
    fr.onload = () => {
      this.codeViews.toArray()[index].nativeElement.value = fr.result
      this.code[index] = fr.result as string
    }
    fr.readAsText(e.target.files[0])
  }

  async sendCode(): Promise<void> {
    let request: Response = await fetch('http://demo.codehonestly.com:5000/ast2json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `pysrc=${encodeURIComponent(this.code[0])}&normalize=true&ctx=false`
    })
    let response: PreprocessingServerResponse = await request.json()
    this.vt.data(response['graph']).update()

    this.imports.clear()
    for (let [key, item] of Object.entries(response['imports'])) {
      this.imports.clear()
      this.imports.set(key, <string>item)
    }
  }

  toggleVisualization(): void {
    this.visualizeMode = !this.visualizeMode
    if (this.visualizeMode) {
      setTimeout(this.ngAfterViewInit.bind(this), 0)
    }
  }

  // selectionChange(e: StepperSelectionEvent): void {
  //   if (e.selectedIndex === 2) {
  //     this.sendCode()
  //   }
  // }
}

export interface PreprocessingServerResponse {
  imports: object
  graph: object
}
