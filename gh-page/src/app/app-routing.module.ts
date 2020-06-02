import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { AppComponent } from './app.component';
import { VisualizeRootComponent } from './visualize-root/visualize-root.component';
import { SimilarityRootComponent } from './similarity-root/similarity-root.component';


const routes: Routes = [
  // {
  //   path: '',
  //   component: AppComponent
  // },
  {
    path: 'visualize',
    component: VisualizeRootComponent
  },
  {
    path: 'similarity',
    component: SimilarityRootComponent
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
