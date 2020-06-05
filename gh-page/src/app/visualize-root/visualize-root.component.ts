import { Component } from '@angular/core';
import { APIs } from '../api/api';

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
    this.graphData = (await APIs.getVisualization(code)).graph
  }
}
