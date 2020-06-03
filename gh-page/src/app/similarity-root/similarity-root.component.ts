import { Component } from '@angular/core';
import { Result } from '../output-results/output-results.component';

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

  async getReport(code: string[]): Promise<void> {
    this.loading = true
    let req = await fetch('http://demo.codehonestly.com:5000/simreport', {
      method: 'POST', headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `pysrc1=${encodeURIComponent(code[0])}&pysrc2=${encodeURIComponent(code[1])}`
    })
    this.loading = false
    this.report = await req.json()
  }

  sendToDetailedView(files: string[]): void {
    [this.file1, this.file2] = files
  }
}
