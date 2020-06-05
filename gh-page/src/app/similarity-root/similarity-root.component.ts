import { Component } from '@angular/core';
import { Result } from '../output-results/output-results.component';
import { APIs } from '../api/api';
import { MatDialog } from '@angular/material/dialog';
import { CompareDialogComponent } from '../compare-dialog/compare-dialog.component';

@Component({
  selector: 'app-similarity-root',
  templateUrl: './similarity-root.component.html',
  styleUrls: ['./similarity-root.component.sass']
})
export class SimilarityRootComponent {
  report: Result = {}
  visualizeMode: boolean = false
  loading: boolean = false
  file1: string = null
  file2: string = null

  _code: string[]

  constructor(private dialog: MatDialog) {}

  async getReport(code: string[]): Promise<void> {
    this._code = Array.from(code)
    this.loading = true
    this.report = await APIs.getReport(code)
    this.loading = false
  }

  sendToDetailedView(files: string[]): void {
    [this.file1, this.file2] = files
  }

  showModal(functions: any): void {
    if (this.file1 === 'input2') {
      this._code = this._code.reverse()
    }
    let fns: string[] = []
    for (let [i, fn] of functions.entries()) {
      let currentFn: string[] = []
      let splitted: string[] = this._code[i].split('\n')
      let hasStart: boolean = false
      for (let line of splitted) {
        if (line.startsWith(`def ${fn}`)) {
          currentFn.push(line)
          hasStart = true
          continue
        }

        if (hasStart) {
          if (line.trimLeft() === line && line.trim() !== '' && line.trim().charAt(0) !== '#') {
            fns.push(currentFn.join('\n'))
            currentFn = []
            break
          } else {
            if (line.trim().charAt(0) !== '#') {
              currentFn.push(line)
            }
          }
        }
      }
      if (currentFn.length !== 0) {
        fns.push(currentFn.join('\n'))
      }
    }

    this.dialog.open(CompareDialogComponent, {
      width: '60%',
      data: fns
    })
    // console.log(fns)
    if (this.file1 === 'input2') {
      this._code = this._code.reverse()
    }
  }
}
