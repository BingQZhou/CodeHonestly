import { Component, Input, AfterViewInit, EventEmitter, Output, ViewChild, ElementRef } from '@angular/core'

declare var vtree: any

@Component({
  selector: 'output-results',
  templateUrl: './output-results.component.html',
  styleUrls: ['./output-results.component.sass']
})
export class OutputResultsComponent implements AfterViewInit {
  _visualizeMode: boolean = true
  _report: Result = {}
  _columns: string[] = ['input1', 'input2']

  @ViewChild('svg') svgElement: ElementRef

  @Input() set visualizeMode(value: boolean) {
    this._visualizeMode = value
    if (this._visualizeMode) {
      this.report = {}
      setTimeout(this.ngAfterViewInit.bind(this), 0)
    }
  }
  @Input() set report(value: Result) {
    this._report = value
    this.displayedColumns = ['name'].concat(this._columns)
  }
  @Input() set visualization(value: object) {
    if (this._visualizeMode) {
      this.vt && this.vt.data(value).update()
    }
  }

  @Input() set numberOfFiles(value: number) {
    this._columns = Array(value).fill(0).map((_, i) => 'input' + String(i + 1))
  }

  @Output() onSimilarityClick: EventEmitter<string[]> = new EventEmitter<string[]>()

  vt: any
  displayedColumns = ['name']

  ngAfterViewInit(): void {
    if (this._visualizeMode) {
      this.vt = vtree(document.getElementById('container')).conf('maxNameLen', 32).conf('maxValueLen', 32)
      this.vt.mode(this.vt.MODE_PYTHON_AST).conf('showArrayNode', false)
      document.querySelector('svg').setAttribute('width', '90vw')
      document.querySelector('svg').setAttribute('height', '90vh')
      document.querySelector('svg').style.boxShadow = 'none'
      this.vt.height = document.querySelector('svg').clientHeight
      this.vt.width = document.querySelector('svg').clientWidth
    }
  }

  cellClick(i: number, j: number): void {
    if (i !== j) {
      this.onSimilarityClick.emit([this._report.overview.rows[i], this._report.overview.columns[j]])
    } else {
      this.onSimilarityClick.emit(['', ''])
    }
  }
}

export interface Result {
  overview?: ResultOverview,
  detailed?: object
  errors?: string[]
}

export interface ResultOverview {
  data: number[][],
  rows: string[],
  columns: string[]
}
