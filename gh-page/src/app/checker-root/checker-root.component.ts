import { Component } from '@angular/core';
import { APIs } from '../api/api';
import { Result } from '../output-results/output-results.component';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { PasswordDialogComponent } from '../password-dialog/password-dialog.component';
import { MatSnackBar } from '@angular/material/snack-bar';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'checker-root',
  templateUrl: './checker-root.component.html',
  styleUrls: ['./checker-root.component.sass']
})
export class CheckerRootComponent {
  report: Result = {}
  loading: boolean = false
  file1: string = null
  file2: string = null
  numberOfFiles: number = 0

  constructor(private dialog: MatDialog, private _snackBar: MatSnackBar) {}
  async sendCode(codeArr: string[]): Promise<void> {
    let dialogRef: MatDialogRef<PasswordDialogComponent, any>
    let password: string
    if (environment.usePassword) {
      dialogRef = this.dialog.open(PasswordDialogComponent, {
        width: '250px',
        data: localStorage.getItem('password')
      })
      password = await dialogRef.afterClosed().toPromise()
      if (password === undefined) {
        return
      }
      localStorage.setItem('password', password)
    } else {
      password = environment.password
    }

    this.numberOfFiles = codeArr.length
    this.loading = true
    try {
      this.report = await APIs.getReport(codeArr, password)
    } catch (e) {
      this._snackBar.open(e.message, 'Dismiss', {
        duration: 5000
      })
    } finally {
      this.loading = false
    }
  }

  sendToDetailedView(files: string[]): void {
    [this.file1, this.file2] = files
  }
}
