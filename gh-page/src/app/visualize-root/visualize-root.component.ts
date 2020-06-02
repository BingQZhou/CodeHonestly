import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-visualize-root',
  templateUrl: './visualize-root.component.html',
  styleUrls: ['./visualize-root.component.sass']
})
export class VisualizeRootComponent {
  visualizeMode: boolean = true
  loading: boolean = false
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
  }
}

export interface PreprocessingServerResponse {
  imports: object
  graph: object
}
