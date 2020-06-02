import { Component } from '@angular/core';
import { Result } from './output-results/output-results.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent {
  report: Result = {}
  visualizeMode: boolean = true
  loading: boolean = false
  imports: Map<string, string> = new Map<string, string>()
  graphData: object = {}

  async sendCode(code: string): Promise<void> {
    let request: Response = await fetch('http://demo.codehonestly.com:5000/ast2json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `pysrc=${encodeURIComponent(code)}&normalize=true&ctx=false`
    })
    let response: PreprocessingServerResponse = await request.json()

    this.graphData = response['graph']

    this.imports.clear()
    for (let [key, item] of Object.entries(response['imports'])) {
      this.imports.clear()
      this.imports.set(key, <string>item)
    }
  }

  toggleVisualization(mode: boolean): void {
    this.visualizeMode = mode
    if (this.visualizeMode) {
      this.report = {}
    }
  }

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
}

export interface PreprocessingServerResponse {
  imports: object
  graph: object
}
