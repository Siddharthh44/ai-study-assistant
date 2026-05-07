import { createBrowserRouter, redirect } from 'react-router';
import { DashboardLayout } from './components/DashboardLayout';
import { AuthPage } from './pages/Auth';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Processing } from './pages/Processing';
import { NotesPage } from './pages/NotesPage';
import { NoteView } from './pages/NoteView';
import { Flashcards } from './pages/Flashcards';
import { Quizzes } from './pages/Quizzes';
import { Quiz } from './pages/Quiz';
import { QuizResults } from './pages/QuizResults';
import { Progress } from './pages/Progress';
import { PYQAnalysis } from './pages/PYQAnalysis';
import { Bookmarks } from './pages/Bookmarks';
import { History } from './pages/History';
import { ExportPage } from './pages/Export';
import { Settings } from './pages/Settings';

export const router = createBrowserRouter([
  {
    path: '/',
    loader: () => redirect('/login'),
  },
  {
    path: '/login',
    Component: AuthPage,
  },
  {
    path: '/processing',
    Component: Processing,
  },
  {
    path: '/app',
    Component: DashboardLayout,
    children: [
      { index: true, Component: Dashboard },
      { path: 'upload', Component: Upload },
      { path: 'notes', Component: NotesPage },
      { path: 'notes/:id', Component: NoteView },
      { path: 'flashcards', Component: Flashcards },
      { path: 'quizzes', Component: Quizzes },
      { path: 'quizzes/start', Component: Quiz },
      { path: 'quizzes/results', Component: QuizResults },
      { path: 'progress', Component: Progress },
      { path: 'pyq-analysis', Component: PYQAnalysis },
      { path: 'bookmarks', Component: Bookmarks },
      { path: 'history', Component: History },
      { path: 'export', Component: ExportPage },
      { path: 'settings', Component: Settings },
    ],
  },
]);
