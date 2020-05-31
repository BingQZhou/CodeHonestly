import { Component, Input, AfterViewInit } from '@angular/core'

declare var vtree: any

@Component({
  selector: 'output-results',
  templateUrl: './output-results.component.html',
  styleUrls: ['./output-results.component.sass']
})
export class OutputResultsComponent implements AfterViewInit {
  _visualizeMode: boolean = true
  @Input() set visualizeMode(value: boolean) {
    this._visualizeMode = value
    if (this._visualizeMode) {
      this.report = {}
      setTimeout(this.ngAfterViewInit.bind(this), 0)
    }
  }
  @Input() report: Result = {}
  @Input() set visualization(value: object) {
    if (this._visualizeMode) {
      this.vt && this.vt.data(value).update()
    }
  }

  vt: any
  displayedColumns = ['func1', 'func2', 'sim']

  ngAfterViewInit(): void {
    this.vt = vtree(document.getElementById('container')).conf('maxNameLen', 32).conf('maxValueLen', 32)
    this.vt.mode(this.vt.MODE_PYTHON_AST).conf('showArrayNode', false)
    document.querySelector('svg').setAttribute('width', '90vw')
    document.querySelector('svg').setAttribute('height', '90vh')
    document.querySelector('svg').style.boxShadow = 'none'
    this.vt.height = document.querySelector('svg').clientHeight
    this.vt.width = document.querySelector('svg').clientWidth
  }
}

export interface Result {
  overall?: number
  pairs?: Array<Array<number>>
  error?: boolean
}
