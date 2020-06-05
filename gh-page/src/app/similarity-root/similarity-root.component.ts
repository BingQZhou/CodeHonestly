import { Component } from '@angular/core';
import { Result } from '../output-results/output-results.component';
import { APIs } from '../api/api';

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
    this.report = await APIs.getReport(code)
    this.loading = false
  }

  sendToDetailedView(files: string[]): void {
    [this.file1, this.file2] = files
  }
}
